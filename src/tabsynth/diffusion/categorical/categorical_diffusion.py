"""Discrete Multinomial Denoising Diffusion Probabilistic Model (D3PM) for categorical features.

Mathematical formulation:
    Forward process (row-vector convention):
        q(x_t | x_0) = x_0 @ Q_bar_t = alpha_bar_t * x_0 + (1 - alpha_bar_t) / K * 1
        
    Exact reverse posterior given clean state x_0:
        q(x_{t-1} | x_t, x_0) propto q(x_t | x_{t-1}) * q(x_{t-1} | x_0)
        q(x_{t-1} | x_t, x_0) propto [alpha_t * x_t + (1 - alpha_t) / K] * [alpha_bar_{t-1} * x_0 + (1 - alpha_bar_{t-1}) / K]
        
    Learned reverse posterior via probability marginalization:
        p_theta(x_{t-1} | x_t) = sum_{k=1}^K q(x_{t-1} | x_t, x_0 = e_k) * p_theta(x_0 = e_k | x_t)
        where p_theta(x_0 = e_k | x_t) = softmax(pred_logits)_k.
        
    Variational KL training objective:
        L_cat = D_KL(q(x_{t-1} | x_t, x_0) || p_theta(x_{t-1} | x_t))   for t > 0
        L_cat = -log p_theta(x_0 | x_0)                                  for t = 0
"""

from typing import List, Tuple, Optional, Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from tabsynth.diffusion.schedules import get_noise_schedule
from tabsynth.diffusion.categorical.transition import (
    get_transition_matrix,
    get_cumulative_transition_matrix,
    validate_transition_matrix,
)
from tabsynth.diffusion.utils import select_device, check_probability_simplex, to_one_hot


class CategoricalDiffusion(nn.Module):
    """Multinomial discrete diffusion engine supporting multiple categorical columns."""

    def __init__(
        self,
        cardinalities: List[int],
        num_timesteps: int = 100,
        schedule_type: Literal["cosine", "linear"] = "cosine",
        device: Optional[Union[str, torch.device]] = "auto",
        **schedule_kwargs,
    ):
        super().__init__()
        assert len(cardinalities) > 0, "cardinalities list must not be empty."
        assert all(k >= 2 for k in cardinalities), f"Each cardinality must be >= 2, got {cardinalities}"

        self.cardinalities = list(cardinalities)
        self.num_cats = len(cardinalities)
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        self.device = select_device(device)

        betas = get_noise_schedule(
            schedule_type=schedule_type,
            timesteps=num_timesteps,
            **schedule_kwargs,
        )
        self.betas = betas.to(device=self.device, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=self.device, dtype=torch.float32), self.alphas_cumprod[:-1]]
        )

    def to_device(self, device: Union[str, torch.device]) -> "CategoricalDiffusion":
        """Moves schedule buffers to specified device."""
        self.device = select_device(device)
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)
        return self

    def q_probs_single_feature(
        self,
        x_0_onehot: torch.Tensor,
        timesteps: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Computes forward categorical transition probabilities q(x_t | x_0) for one feature.
        
        q(x_t | x_0) = alpha_bar_t * x_0 + (1 - alpha_bar_t) / K * 1
        
        Args:
            x_0_onehot: Clean one-hot tensor (B, K).
            timesteps: 1D tensor of timesteps (B,).
            K: Cardinality of feature.
            
        Returns:
            probs: Transition probability tensor of shape (B, K) summing to 1.
        """
        assert x_0_onehot.ndim == 2 and x_0_onehot.shape[-1] == K
        alpha_bar_t = self.alphas_cumprod.gather(0, timesteps.to(self.device)).unsqueeze(-1)  # (B, 1)
        probs = alpha_bar_t * x_0_onehot + (1.0 - alpha_bar_t) / float(K)
        return torch.clamp(probs, min=1e-12, max=1.0)

    def q_sample_single_feature(
        self,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        K: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Corrupts a single categorical feature by sampling from q(x_t | x_0).
        
        Supports both integer category IDs (B,) and one-hot tensors (B, K).
        
        Returns:
            x_t_onehot: Corrupted one-hot tensor (B, K).
            x_t_indices: Corrupted integer indices (B,).
        """
        if x_0.ndim == 1 or (x_0.ndim == 2 and x_0.shape[-1] == 1):
            x_0_onehot = to_one_hot(x_0.squeeze(-1).long(), num_classes=K).to(self.device)
        else:
            x_0_onehot = x_0.to(device=self.device, dtype=torch.float32)

        probs = self.q_probs_single_feature(x_0_onehot=x_0_onehot, timesteps=timesteps, K=K)
        # Sample discrete categorical index from transition probability distribution
        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
        x_t_onehot = to_one_hot(samples, num_classes=K)
        return x_t_onehot, samples

    def q_sample_categorical(
        self,
        x_0_list: List[torch.Tensor],
        timesteps: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Corrupts a list of categorical features across all columns.
        
        Args:
            x_0_list: List of tensors, one per categorical column.
            timesteps: 1D tensor of timesteps (B,).
            
        Returns:
            x_t_onehots: List of corrupted one-hot tensors of shapes (B, K_j).
            x_t_indices: List of corrupted integer index tensors of shapes (B,).
        """
        assert len(x_0_list) == self.num_cats, (
            f"Expected {self.num_cats} categorical features, got {len(x_0_list)}"
        )
        x_t_onehots = []
        x_t_indices = []

        for j, (x_0, K) in enumerate(zip(x_0_list, self.cardinalities)):
            onehot, idx = self.q_sample_single_feature(x_0=x_0, timesteps=timesteps, K=K)
            x_t_onehots.append(onehot)
            x_t_indices.append(idx)

        return x_t_onehots, x_t_indices

    def q_posterior_probs(
        self,
        x_t_onehot: torch.Tensor,
        x_0_onehot: torch.Tensor,
        timesteps: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Calculates exact discrete reverse posterior q(x_{t-1} | x_t, x_0).
        
        By Bayes rule:
            q(x_{t-1} | x_t, x_0) propto q(x_t | x_{t-1}) * q(x_{t-1} | x_0)
            
        In row-vector notation:
            q(x_t | x_{t-1}) = alpha_t * x_t + (1 - alpha_t) / K * 1
            q(x_{t-1} | x_0) = alpha_bar_{t-1} * x_0 + (1 - alpha_bar_{t-1}) / K * 1
            
        Args:
            x_t_onehot: Noisy state one-hot tensor (B, K).
            x_0_onehot: Clean state one-hot tensor (B, K).
            timesteps: 1D timesteps tensor (B,).
            K: Number of categories.
            
        Returns:
            posterior_probs: (B, K) normalized probability distribution over x_{t-1}.
        """
        alpha_t = self.alphas.gather(0, timesteps.to(self.device)).unsqueeze(-1)  # (B, 1)
        alpha_bar_prev = self.alphas_cumprod_prev.gather(0, timesteps.to(self.device)).unsqueeze(-1)  # (B, 1)

        # q(x_t | x_{t-1})
        fact_t = alpha_t * x_t_onehot + (1.0 - alpha_t) / float(K)
        # q(x_{t-1} | x_0)
        fact_prev = alpha_bar_prev * x_0_onehot + (1.0 - alpha_bar_prev) / float(K)

        unnorm = fact_t * fact_prev
        norm = unnorm.sum(dim=-1, keepdim=True)
        return unnorm / torch.clamp(norm, min=1e-12)

    def p_probs(
        self,
        x_t_onehot: torch.Tensor,
        pred_x_0_logits: torch.Tensor,
        timesteps: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Calculates learned reverse distribution p_theta(x_{t-1} | x_t) via probability marginalization.
        
        p_theta(x_{t-1} | x_t) = sum_{k=1}^K q(x_{t-1} | x_t, x_0 = e_k) * p_theta(x_0 = e_k | x_t)
        
        CRITICAL: Never takes argmax over predicted categories. Full probability vector is marginalized.
        
        Args:
            x_t_onehot: Current noisy state one-hot tensor (B, K).
            pred_x_0_logits: Model predicted clean logits (B, K).
            timesteps: 1D timesteps tensor (B,).
            K: Number of categories.
            
        Returns:
            p_prev: (B, K) valid probability distribution on the simplex.
        """
        # Predicted clean probabilities: p_theta(x_0 = e_k | x_t)
        pred_x_0_probs = F.softmax(pred_x_0_logits, dim=-1)  # (B, K)

        batch_size = x_t_onehot.shape[0]
        identity_k = torch.eye(K, device=self.device, dtype=torch.float32)  # (K, K)

        # Expand across all K basis states:
        # x_t: (B, K) -> (B * K, K)
        # t: (B,) -> (B * K,)
        # e_k: (K, K) -> (B * K, K)
        x_t_expanded = x_t_onehot.unsqueeze(1).expand(-1, K, -1).reshape(batch_size * K, K)
        t_expanded = timesteps.unsqueeze(1).expand(-1, K).reshape(batch_size * K)
        e_k_expanded = identity_k.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * K, K)

        # Evaluate exact posterior for each basis vector e_k:
        # Result shape: (B, K, K) where [b, k, :] is q(x_{t-1} | x_t, x_0 = e_k)
        q_post_all_k = self.q_posterior_probs(
            x_t_onehot=x_t_expanded,
            x_0_onehot=e_k_expanded,
            timesteps=t_expanded,
            K=K,
        ).reshape(batch_size, K, K)

        # Marginalize: (B, 1, K) @ (B, K, K) -> (B, 1, K) -> (B, K)
        p_prev = torch.bmm(pred_x_0_probs.unsqueeze(1), q_post_all_k).squeeze(1)
        p_prev = torch.clamp(p_prev, min=1e-12, max=1.0)
        # Renormalize to ensure exact simplex membership
        return p_prev / p_prev.sum(dim=-1, keepdim=True)

    def compute_kl_divergence(
        self,
        x_t_onehot: torch.Tensor,
        x_0_onehot: torch.Tensor,
        pred_x_0_logits: torch.Tensor,
        timesteps: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Computes exact variational KL divergence objective for a single feature.
        
        L_cat = D_KL(q(x_{t-1} | x_t, x_0) || p_theta(x_{t-1} | x_t))   for t > 0
        L_cat = -log p_theta(x_0 | x_0)                                  for t = 0
        
        Args:
            x_t_onehot: (B, K)
            x_0_onehot: (B, K)
            pred_x_0_logits: (B, K)
            timesteps: (B,)
            K: Cardinality
            
        Returns:
            Scalar mean loss across the batch.
        """
        # True posterior distribution
        q_post = self.q_posterior_probs(
            x_t_onehot=x_t_onehot,
            x_0_onehot=x_0_onehot,
            timesteps=timesteps,
            K=K,
        )
        # Learned reverse distribution
        p_pred = self.p_probs(
            x_t_onehot=x_t_onehot,
            pred_x_0_logits=pred_x_0_logits,
            timesteps=timesteps,
            K=K,
        )

        # D_KL = sum_k q_k * (log(q_k) - log(p_k))
        kl = q_post * (torch.log(torch.clamp(q_post, min=1e-12)) - torch.log(torch.clamp(p_pred, min=1e-12)))
        kl_per_sample = kl.sum(dim=-1)

        # At t = 0: exact negative log-likelihood of true clean state x_0
        pred_log_probs = F.log_softmax(pred_x_0_logits, dim=-1)
        nll_at_zero = -(x_0_onehot * pred_log_probs).sum(dim=-1)

        loss = torch.where(timesteps == 0, nll_at_zero, kl_per_sample)
        return loss.mean()

    def auxiliary_cross_entropy_loss(
        self,
        pred_x_0_logits: torch.Tensor,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """Auxiliary direct clean-category cross-entropy regularizer.
        
        Kept separate from the variational D3PM KL objective.
        """
        if x_0.ndim == 2 and x_0.shape[-1] > 1:
            targets = torch.argmax(x_0, dim=-1)
        else:
            targets = x_0.squeeze(-1).long()
        return F.cross_entropy(pred_x_0_logits, targets)
