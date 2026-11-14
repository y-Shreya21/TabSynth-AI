"""Integration module connecting denoising neural models with the Phase 02 diffusion engine."""

from typing import List, Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from tabsynth.diffusion.mixed.mixed_diffusion import MixedTypeDiffusion
from tabsynth.diffusion.utils import select_device, to_one_hot


class ModelDiffusionIntegrator(nn.Module):
    """Bridges TabSynth neural denoisers with the MixedTypeDiffusion mathematical engine.
    
    Orchestrates:
    1. Forward noising: q(x_t^num | x_0^num) and q(x_t^cat | x_0^cat).
    2. Neural prediction: model(x_t^num, x_t^cat, t_num, t_cat, class_labels).
    3. Composite loss evaluation:
       L_total = lambda_num * L_num (MSE) + lambda_cat * L_cat (D3PM KL) + lambda_aux * L_aux (CE).
    """

    def __init__(
        self,
        model: nn.Module,
        diffusion: MixedTypeDiffusion,
    ):
        super().__init__()
        self.model = model
        self.diffusion = diffusion

    def forward_loss(
        self,
        x_num_0: Optional[torch.Tensor] = None,
        x_cat_0_list: Optional[List[torch.Tensor]] = None,
        class_labels: Optional[torch.Tensor] = None,
        t_num: Optional[torch.Tensor] = None,
        t_cat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Runs forward diffusion corruption, neural denoising, and multi-objective loss calculation.
        
        Args:
            x_num_0: Clean continuous features (B, D_num) or None.
            x_cat_0_list: Clean categorical features [(B, K_j)] or None.
            class_labels: Target class labels (B,) or None.
            t_num: Optional specified continuous timesteps (B,).
            t_cat: Optional specified discrete timesteps (B,).
            
        Returns:
            losses_dict containing:
                "loss_total": Weighted composite scalar loss.
                "loss_num": Continuous noise prediction MSE loss.
                "loss_cat": Discrete categorical variational KL loss.
                "loss_aux": Auxiliary clean-state cross-entropy loss.
        """
        device = self.diffusion.device

        # Determine batch size
        if x_num_0 is not None and x_num_0.shape[1] > 0:
            batch_size = x_num_0.shape[0]
            dev = x_num_0.device
        elif x_cat_0_list is not None and len(x_cat_0_list) > 0:
            batch_size = x_cat_0_list[0].shape[0]
            dev = x_cat_0_list[0].device
        else:
            raise ValueError("Must provide at least one continuous or categorical feature batch.")

        # 1. Sample Timesteps if not provided
        if t_num is None and self.diffusion.num_numerical_features > 0:
            t_num = torch.randint(0, self.diffusion.numerical_timesteps, (batch_size,), device=dev)
        if t_cat is None and self.diffusion.num_categorical_features > 0:
            t_cat = torch.randint(0, self.diffusion.categorical_timesteps, (batch_size,), device=dev)

        # 2. Forward Diffusion Corruption
        x_t_num, target_noise, x_t_cat_oh, _ = self.diffusion.forward(
            x_num=x_num_0,
            x_cat_list=x_cat_0_list,
            t_num=t_num,
            t_cat=t_cat,
        )

        # 3. Model Forward Pass
        output = self.model(
            x_num_t=x_t_num,
            x_cat_t=x_t_cat_oh,
            timesteps_num=t_num,
            timesteps_cat=t_cat,
            class_labels=class_labels,
        )
        pred_eps = output["predicted_noise"]
        pred_logits = output["categorical_logits"]

        # 4. Compute Multi-Objective Diffusion Losses
        losses = self.diffusion.compute_training_loss(
            pred_eps=pred_eps,
            target_noise=target_noise,
            pred_logits_list=pred_logits,
            clean_cat_list=x_cat_0_list,
            corrupted_cat_list=x_t_cat_oh,
            t_cat=t_cat,
        )

        return losses

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        class_labels: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generates synthetic records through ancestral reverse diffusion using the trained model."""
        def model_wrapper(x_n, x_c, t_n, t_c):
            out = self.model(
                x_num_t=x_n,
                x_cat_t=x_c,
                timesteps_num=t_n,
                timesteps_cat=t_c,
                class_labels=class_labels,
            )
            return out["predicted_noise"], out["categorical_logits"]

        return self.diffusion.sample(
            model_fn=model_wrapper,
            batch_size=batch_size,
            device=self.diffusion.device,
            clip_denoised=clip_denoised,
        )
