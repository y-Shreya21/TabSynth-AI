"""Iterative discrete ancestral reverse sampler for categorical features."""

from typing import Callable, List, Tuple, Optional, Union
import torch
import torch.nn.functional as F

from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.diffusion.utils import select_device, to_one_hot


class CategoricalSampler:
    """Ancestral reverse sampler for discrete categorical features using D3PM posterior transitions."""

    def __init__(self, diffusion: CategoricalDiffusion):
        self.diffusion = diffusion
        self.cardinalities = diffusion.cardinalities
        self.num_cats = diffusion.num_cats
        self.num_timesteps = diffusion.num_timesteps

    @torch.no_grad()
    def sample(
        self,
        model_fn: Callable[[List[torch.Tensor], torch.Tensor], List[torch.Tensor]],
        batch_size: int,
        device: Optional[Union[str, torch.device]] = None,
        return_indices: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Samples categorical features across all columns from pure uniform noise.
        
        Args:
            model_fn: Callable taking (x_cat_list, t) and returning list of logits [(B, K_1), ...].
            batch_size: Number of synthetic rows B to generate.
            device: Target device.
            return_indices: If True, also returns list of integer class index tensors.
            
        Returns:
            x_cat_onehots: List of one-hot tensors [(B, K_1), ..., (B, K_C)].
            (Optional) x_cat_indices: List of integer class index tensors [(B,), ..., (B,)].
        """
        dev = select_device(device) if device is not None else self.diffusion.device

        # Step 1: Initialize from uniform categorical distribution at T-1
        current_onehots: List[torch.Tensor] = []
        for K in self.cardinalities:
            uniform_probs = torch.full((batch_size, K), 1.0 / float(K), device=dev, dtype=torch.float32)
            init_idx = torch.multinomial(uniform_probs, num_samples=1).squeeze(-1)
            current_onehots.append(to_one_hot(init_idx, num_classes=K))

        # Step 2: Reverse iterative loop from T-1 down to 0
        for t_idx in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t_idx, device=dev, dtype=torch.long)
            # Model predicts clean logits for each categorical column
            predicted_logits_list = model_fn(current_onehots, t_batch)

            next_onehots: List[torch.Tensor] = []
            for j, K in enumerate(self.cardinalities):
                x_t = current_onehots[j]
                pred_logits = predicted_logits_list[j]

                if t_idx == 0:
                    # Final step: argmax over predicted logits or sample from softmax
                    clean_indices = torch.argmax(pred_logits, dim=-1)
                    x_prev = to_one_hot(clean_indices, num_classes=K)
                else:
                    # Discrete reverse step via marginalized posterior p_theta(x_{t-1} | x_t)
                    p_prev = self.diffusion.p_probs(
                        x_t_onehot=x_t,
                        pred_x_0_logits=pred_logits,
                        timesteps=t_batch,
                        K=K,
                    )
                    samples = torch.multinomial(p_prev, num_samples=1).squeeze(-1)
                    x_prev = to_one_hot(samples, num_classes=K)

                next_onehots.append(x_prev)

            current_onehots = next_onehots

        if return_indices:
            indices_list = [torch.argmax(oh, dim=-1) for oh in current_onehots]
            return current_onehots, indices_list

        return current_onehots
