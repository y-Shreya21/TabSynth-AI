"""Joint reverse ancestral sampler for continuous Gaussian and discrete D3PM processes."""

from typing import Optional, Dict, List, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from tabsynth.diffusion.utils import select_device, to_one_hot
from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.diffusion.mixed.mixed_diffusion import MixedTypeDiffusion
from tabsynth.sampling.guidance import apply_continuous_cfg, apply_categorical_cfg


class MixedAncestralSampler:
    """Ancestral reverse diffusion sampler coordinating continuous Gaussian and discrete D3PM processes."""

    def __init__(
        self,
        model: nn.Module,
        num_continuous: int,
        cardinalities: Optional[List[int]] = None,
        gaussian_diffusion: Optional[GaussianDiffusion] = None,
        multinomial_diffusion: Optional[CategoricalDiffusion] = None,
        mixed_diffusion: Optional[MixedTypeDiffusion] = None,
        num_timesteps: int = 100,
        device: Union[str, torch.device] = "auto",
    ):
        self.device = select_device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.num_continuous = num_continuous
        self.cardinalities = list(cardinalities or [])
        self.num_timesteps = num_timesteps

        self.gaussian_diff = gaussian_diffusion
        if self.gaussian_diff is not None and hasattr(self.gaussian_diff, "to_device"):
            self.gaussian_diff.to_device(self.device)
            self.num_timesteps = self.gaussian_diff.num_timesteps

        self.multinomial_diff = multinomial_diffusion
        if self.multinomial_diff is not None and hasattr(self.multinomial_diff, "to_device"):
            self.multinomial_diff.to_device(self.device)

        self.mixed_diff = mixed_diffusion
        if self.mixed_diff is not None and hasattr(self.mixed_diff, "to_device"):
            self.mixed_diff.to_device(self.device)
            self.num_timesteps = self.mixed_diff.numerical_timesteps

    @torch.no_grad()
    def sample_step(
        self,
        batch_size: int,
        c: Optional[torch.Tensor] = None,
        cfg_scale: float = 0.0,
        clip_denoised: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Executes full reverse diffusion trajectory from t=T-1 to t=0.
        
        Args:
            batch_size: Number of synthetic instances to sample.
            c: Optional conditioning tensor (e.g. target class labels) of shape (B,).
            cfg_scale: Classifier-Free Guidance scale w.
            clip_denoised: Whether to clamp continuous predictions to prevent explosive tails.
            
        Returns:
            x_num: Continuous synthetic features (batch_size, num_continuous).
            x_cat_list: List of one-hot discrete features [(batch_size, K_j)].
        """
        self.model.eval()

        # 1. Initialize Continuous Prior: x_T ~ N(0, I)
        if self.num_continuous > 0:
            x_num_t = torch.randn(batch_size, self.num_continuous, device=self.device, dtype=torch.float32)
        else:
            x_num_t = torch.empty(batch_size, 0, device=self.device, dtype=torch.float32)

        # 2. Initialize Discrete Prior: x_T ~ Uniform(1..K_j)
        x_cat_t_list = []
        for K in self.cardinalities:
            uniform = torch.full((batch_size, K), 1.0 / float(K), device=self.device, dtype=torch.float32)
            idx = torch.multinomial(uniform, num_samples=1).squeeze(-1)
            x_cat_t_list.append(to_one_hot(idx, num_classes=K))

        # 3. Iterative Reverse Diffusion Loop: T-1 -> 0
        steps = self.num_timesteps
        for step in reversed(range(steps)):
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)

            # Conditional model forward pass
            out_cond = self.model(
                x_num_t=x_num_t,
                x_cat_t=x_cat_t_list,
                timesteps_num=t,
                timesteps_cat=t,
                class_labels=c,
                force_uncond=False,
            )

            if isinstance(out_cond, tuple):
                pred_eps_cond, pred_cat_logits_cond = out_cond
            elif isinstance(out_cond, dict):
                pred_eps_cond = out_cond["predicted_noise"]
                pred_cat_logits_cond = out_cond["categorical_logits"]
            else:
                pred_eps_cond = out_cond.predicted_noise
                pred_cat_logits_cond = out_cond.categorical_logits

            # Classifier-Free Guidance extrapolation
            if cfg_scale > 0.0 and c is not None:
                out_uncond = self.model(
                    x_num_t=x_num_t,
                    x_cat_t=x_cat_t_list,
                    timesteps_num=t,
                    timesteps_cat=t,
                    class_labels=c,
                    force_uncond=True,
                )
                if isinstance(out_uncond, tuple):
                    pred_eps_uncond, pred_cat_logits_uncond = out_uncond
                elif isinstance(out_uncond, dict):
                    pred_eps_uncond = out_uncond["predicted_noise"]
                    pred_cat_logits_uncond = out_uncond["categorical_logits"]
                else:
                    pred_eps_uncond = out_uncond.predicted_noise
                    pred_cat_logits_uncond = out_uncond.categorical_logits

                pred_eps = apply_continuous_cfg(pred_eps_cond, pred_eps_uncond, cfg_scale)
                pred_cat_logits = apply_categorical_cfg(pred_cat_logits_cond, pred_cat_logits_uncond, cfg_scale)
            else:
                pred_eps = pred_eps_cond
                pred_cat_logits = pred_cat_logits_cond

            # 4. Continuous Reverse Transition Step: x_{t-1}^num
            if self.num_continuous > 0:
                if self.gaussian_diff is not None:
                    if hasattr(self.gaussian_diff, "p_sample_step"):
                        x_num_t = self.gaussian_diff.p_sample_step(
                            predicted_eps=pred_eps,
                            x_t=x_num_t,
                            timesteps=t,
                            clip_denoised=clip_denoised,
                        )
                    elif hasattr(self.gaussian_diff, "p_sample"):
                        x_num_t = self.gaussian_diff.p_sample(
                            predicted_eps=pred_eps,
                            x_t=x_num_t,
                            t=t,
                            clip_denoised=clip_denoised,
                        )
                elif self.mixed_diff is not None and self.mixed_diff.numerical_diffusion is not None:
                    x_num_t = self.mixed_diff.numerical_diffusion.p_sample_step(
                        predicted_eps=pred_eps,
                        x_t=x_num_t,
                        timesteps=t,
                        clip_denoised=clip_denoised,
                    )

            # 5. Discrete Reverse Transition Step: x_{t-1}^cat
            if len(self.cardinalities) > 0:
                next_cat_list = []
                for j, K in enumerate(self.cardinalities):
                    x_t_j = x_cat_t_list[j]
                    logits_j = pred_cat_logits[j]

                    if step == 0:
                        # Final step: collapse to argmax clean prediction
                        clean_idx = torch.argmax(logits_j, dim=-1)
                        x_prev_j = to_one_hot(clean_idx, num_classes=K)
                    else:
                        diff_engine = (
                            self.multinomial_diff
                            if self.multinomial_diff is not None
                            else (self.mixed_diff.categorical_diffusion if self.mixed_diff is not None else None)
                        )
                        if diff_engine is not None:
                            if hasattr(diff_engine, "p_sample"):
                                x_prev_j = diff_engine.p_sample(
                                    x_t_onehot=x_t_j,
                                    pred_x_0_logits=logits_j,
                                    t=t,
                                    K=K,
                                )
                            elif hasattr(diff_engine, "p_probs"):
                                p_prev = diff_engine.p_probs(
                                    x_t_onehot=x_t_j,
                                    pred_x_0_logits=logits_j,
                                    timesteps=t,
                                    K=K,
                                )
                                sample_idx = torch.multinomial(p_prev, num_samples=1).squeeze(-1)
                                x_prev_j = to_one_hot(sample_idx, num_classes=K)
                            else:
                                x_prev_j = x_t_j
                        else:
                            clean_idx = torch.argmax(logits_j, dim=-1)
                            x_prev_j = to_one_hot(clean_idx, num_classes=K)

                    next_cat_list.append(x_prev_j)
                x_cat_t_list = next_cat_list

        return x_num_t, x_cat_t_list
