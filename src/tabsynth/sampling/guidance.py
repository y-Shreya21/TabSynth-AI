"""Classifier-Free Guidance (CFG) extrapolation engine for mixed-type tabular diffusion."""

from typing import List, Tuple, Optional
import torch


def apply_continuous_cfg(
    pred_eps_cond: torch.Tensor,
    pred_eps_uncond: torch.Tensor,
    cfg_scale: float = 0.0,
) -> torch.Tensor:
    """Applies Classifier-Free Guidance extrapolation to continuous noise predictions.
    
    Formula:
        eps_guided = (1 + w) * eps_cond - w * eps_uncond
        
    When w = 0, eps_guided == eps_cond.
    When w > 0, pushes the trajectory away from unconditional density toward conditioned class modes.
    """
    if cfg_scale <= 0.0 or pred_eps_cond.shape[1] == 0:
        return pred_eps_cond

    return (1.0 + cfg_scale) * pred_eps_cond - cfg_scale * pred_eps_uncond


def apply_categorical_cfg(
    pred_logits_cond: List[torch.Tensor],
    pred_logits_uncond: List[torch.Tensor],
    cfg_scale: float = 0.0,
) -> List[torch.Tensor]:
    """Applies Classifier-Free Guidance extrapolation to unnormalized categorical logits.
    
    Formula:
        logits_guided^(j) = (1 + w) * logits_cond^(j) - w * logits_uncond^(j)
    """
    if cfg_scale <= 0.0 or len(pred_logits_cond) == 0:
        return pred_logits_cond

    guided_logits = []
    for l_cond, l_uncond in zip(pred_logits_cond, pred_logits_uncond):
        guided = (1.0 + cfg_scale) * l_cond - cfg_scale * l_uncond
        guided_logits.append(guided)

    return guided_logits


class CFGController:
    """Manages guidance evaluation and extrapolation across heterogeneous tabular heads."""

    def __init__(self, cfg_scale: float = 1.0):
        self.cfg_scale = max(0.0, float(cfg_scale))

    def extrapolate(
        self,
        pred_eps_cond: torch.Tensor,
        pred_eps_uncond: Optional[torch.Tensor],
        pred_cat_cond: List[torch.Tensor],
        pred_cat_uncond: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Extrapolates both continuous noise and discrete logits using configured CFG scale."""
        if self.cfg_scale <= 0.0 or pred_eps_uncond is None or pred_cat_uncond is None:
            return pred_eps_cond, pred_cat_cond

        guided_eps = apply_continuous_cfg(pred_eps_cond, pred_eps_uncond, self.cfg_scale)
        guided_cat = apply_categorical_cfg(pred_cat_cond, pred_cat_uncond, self.cfg_scale)
        return guided_eps, guided_cat
