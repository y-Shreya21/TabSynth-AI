"""Composite loss function balancing Gaussian continuous MSE, discrete D3PM KL, and auxiliary CE."""

from typing import List, Tuple, Dict, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion

# Backward compatibility alias
MultinomialDiffusion = CategoricalDiffusion


class TabSynthLoss(nn.Module):
    """Composite mixed-type diffusion training loss function.
    
    L_total = lambda_num * L_num + lambda_cat * L_cat + lambda_aux * L_aux
    
    Where:
        L_num: Gaussian continuous noise prediction MSE.
        L_cat: Variational lower bound KL divergence over discrete categorical transitions.
        L_aux: Direct cross-entropy loss between predicted clean state logits and x_0.
    """

    def __init__(
        self,
        multinomial_diffusion: Optional[CategoricalDiffusion] = None,
        cardinalities: Optional[List[int]] = None,
        lambda_num: float = 1.0,
        lambda_cat: float = 0.2,
        lambda_aux: float = 0.1,
    ):
        super().__init__()
        self.multinomial_diffusion = multinomial_diffusion
        self.cardinalities = list(cardinalities or [])
        self.lambda_num = lambda_num
        self.lambda_cat = lambda_cat
        self.lambda_aux = lambda_aux

    def forward(
        self,
        pred_eps: torch.Tensor,
        target_eps: torch.Tensor,
        pred_cat_logits: List[torch.Tensor],
        x_cat_t_list: List[torch.Tensor],
        x_cat_0_list: List[torch.Tensor],
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Args:
            pred_eps: Predicted continuous noise (B, N_num).
            target_eps: Ground truth injected noise (B, N_num).
            pred_cat_logits: Predicted clean categorical logits [(B, K_j)].
            x_cat_t_list: Perturbed one-hot categoricals [(B, K_j)].
            x_cat_0_list: Clean one-hot categoricals [(B, K_j)].
            t: Timesteps tensor (B,).

        Returns:
            total_loss: Scalar loss tensor for backpropagation.
            metrics_dict: Dictionary of individual loss components.
        """
        device = t.device if t is not None else pred_eps.device

        # 1. Continuous Gaussian MSE Loss
        if pred_eps is not None and target_eps is not None and pred_eps.shape[1] > 0 and target_eps.shape[1] > 0:
            loss_num = F.mse_loss(pred_eps, target_eps)
        else:
            loss_num = torch.tensor(0.0, device=device, dtype=torch.float32)

        # 2. Discrete D3PM KL Divergence & Auxiliary Cross-Entropy Losses
        loss_cat_terms = []
        loss_aux_terms = []

        if self.cardinalities and len(self.cardinalities) > 0 and pred_cat_logits and len(pred_cat_logits) > 0:
            for idx, K in enumerate(self.cardinalities):
                x_t_j = x_cat_t_list[idx]
                x_0_j = x_cat_0_list[idx]
                logits_j = pred_cat_logits[idx]

                # D3PM Variational KL divergence
                if self.multinomial_diffusion is not None:
                    sig = self.multinomial_diffusion.compute_kl_divergence.__code__.co_varnames
                    if "timesteps" in sig:
                        kl_j = self.multinomial_diffusion.compute_kl_divergence(
                            x_t_onehot=x_t_j,
                            x_0_onehot=x_0_j,
                            pred_x_0_logits=logits_j,
                            timesteps=t,
                            K=K,
                        )
                    else:
                        kl_j = self.multinomial_diffusion.compute_kl_divergence(
                            x_t_onehot=x_t_j,
                            x_0_onehot=x_0_j,
                            pred_x_0_logits=logits_j,
                            t=t,
                            K=K,
                        )
                    loss_cat_terms.append(kl_j)

                # Auxiliary clean cross-entropy
                # x_0_j is one-hot; cross entropy: -sum(x_0 * log_softmax(logits))
                log_p = F.log_softmax(logits_j, dim=-1)
                ce_j = -(x_0_j * log_p).sum(dim=-1).mean()
                loss_aux_terms.append(ce_j)

        if len(loss_cat_terms) > 0:
            loss_cat = torch.stack(loss_cat_terms).mean()
        else:
            loss_cat = torch.tensor(0.0, device=device, dtype=torch.float32)

        if len(loss_aux_terms) > 0:
            loss_aux = torch.stack(loss_aux_terms).mean()
        else:
            loss_aux = torch.tensor(0.0, device=device, dtype=torch.float32)

        # 3. Composite Total Objective
        total_loss = (
            self.lambda_num * loss_num
            + self.lambda_cat * loss_cat
            + self.lambda_aux * loss_aux
        )

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_num_mse": float(loss_num.item()),
            "loss_cat_kl": float(loss_cat.item()),
            "loss_aux_ce": float(loss_aux.item()),
        }

        return total_loss, metrics
