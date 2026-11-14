"""Multinomial Discrete Denoising Diffusion Probabilistic Model (D3PM) for categorical features."""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabsynth.diffusion.schedules import get_noise_schedule


class MultinomialDiffusion(nn.Module):
    """Discrete Multinomial Diffusion Engine based on D3PM (Austin et al., 2021).
    
    Operates directly on categorical distributions via Markovian transition matrices,
    preventing continuous artifacts on discrete categorical states.
    """

    def __init__(
        self,
        cardinalities: List[int],
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
        device: Optional[torch.device] = None,
        **schedule_kwargs,
    ):
        super().__init__()
        self.cardinalities = cardinalities
        self.num_cats = len(cardinalities)
        self.num_timesteps = num_timesteps
        self.device = device or torch.device("cpu")

        betas = get_noise_schedule(schedule_type=schedule_type, timesteps=num_timesteps, **schedule_kwargs)
        self.betas = betas.to(device=self.device, dtype=torch.float32)

        # Precompute alphas and cumulative products: (1 - beta_bar_t) = prod(1 - beta_s)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=self.device, dtype=torch.float32), self.alphas_cumprod[:-1]]
        )

    def to_device(self, device: torch.device) -> "MultinomialDiffusion":
        """Move buffers to specified compute device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        return self

    def q_probs(self, x_0_onehot: torch.Tensor, t: torch.Tensor, K: int) -> torch.Tensor:
        """Forward categorical transition probability vector q(x_t | x_0).
        
        Under uniform categorical transition kernel:
            q(x_t | x_0) = alpha_bar_t * x_0 + (1 - alpha_bar_t) / K * 1
        """
        alpha_bar_t = self.alphas_cumprod[t].unsqueeze(-1)  # (batch_size, 1)
        probs = alpha_bar_t * x_0_onehot + (1.0 - alpha_bar_t) / float(K)
        return torch.clamp(probs, min=1e-12, max=1.0)

    def q_sample(
        self,
        x_0_onehot: torch.Tensor,
        t: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0) on the categorical simplex.
        
        Returns one-hot tensor of shape (batch_size, K).
        """
        probs = self.q_probs(x_0_onehot=x_0_onehot, t=t, K=K)
        # Sample discrete categorical index via Gumbel-Max or multinomial
        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return F.one_hot(samples, num_classes=K).float()

    def q_posterior_probs(
        self,
        x_t_onehot: torch.Tensor,
        x_0_onehot: torch.Tensor,
        t: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Compute exact discrete reverse posterior q(x_{t-1} | x_t, x_0).
        
        q(x_{t-1} | x_t, x_0) propto q(x_t | x_{t-1}) * q(x_{t-1} | x_0)
        """
        alpha_t = self.alphas[t].unsqueeze(-1)
        alpha_bar_prev = self.alphas_cumprod_prev[t].unsqueeze(-1)

        # q(x_t | x_{t-1}) term
        fact1 = alpha_t * x_t_onehot + (1.0 - alpha_t) / float(K)
        # q(x_{t-1} | x_0) term
        fact2 = alpha_bar_prev * x_0_onehot + (1.0 - alpha_bar_prev) / float(K)

        unnorm_probs = fact1 * fact2
        norm = unnorm_probs.sum(dim=-1, keepdim=True)
        return unnorm_probs / torch.clamp(norm, min=1e-12)

    def p_probs(
        self,
        x_t_onehot: torch.Tensor,
        pred_x_0_logits: torch.Tensor,
        t: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Compute model reverse step probability vector p_theta(x_{t-1} | x_t).
        
        Marginalizes over the predicted clean state distribution p_theta(x_0 | x_t):
            p_theta(x_{t-1} | x_t) = sum_k q(x_{t-1} | x_t, x_0 = e_k) * p_theta(x_0 = e_k | x_t)
        """
        pred_x_0_probs = F.softmax(pred_x_0_logits, dim=-1)  # (B, K)

        # Vectorized marginalization over the K basis states:
        # For each k in 0..K-1, evaluate q_posterior with x_0 = e_k
        batch_size = x_t_onehot.shape[0]
        identity_k = torch.eye(K, device=self.device, dtype=torch.float32)  # (K, K)
        
        # Expand x_t and t across K: (B, K, K)
        x_t_expanded = x_t_onehot.unsqueeze(1).expand(-1, K, -1).reshape(batch_size * K, K)
        t_expanded = t.unsqueeze(1).expand(-1, K).reshape(batch_size * K)
        e_k_expanded = identity_k.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * K, K)

        q_post_all_k = self.q_posterior_probs(
            x_t_onehot=x_t_expanded,
            x_0_onehot=e_k_expanded,
            t=t_expanded,
            K=K,
        ).reshape(batch_size, K, K)  # (B, basis_k, x_{t-1}_state)

        # Weighted sum by predicted clean probabilities: (B, K) @ (B, K, K) -> (B, K)
        p_prev = torch.bmm(pred_x_0_probs.unsqueeze(1), q_post_all_k).squeeze(1)
        return torch.clamp(p_prev, min=1e-12, max=1.0)

    def p_sample(
        self,
        x_t_onehot: torch.Tensor,
        pred_x_0_logits: torch.Tensor,
        t: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Sample discrete state x_{t-1} ~ p_theta(x_{t-1} | x_t)."""
        # If t == 0, return argmax or sample directly from predicted clean distribution
        is_t_zero = (t == 0)
        if is_t_zero.all():
            samples = torch.argmax(pred_x_0_logits, dim=-1)
            return F.one_hot(samples, num_classes=K).float()

        p_prev = self.p_probs(x_t_onehot=x_t_onehot, pred_x_0_logits=pred_x_0_logits, t=t, K=K)
        samples = torch.multinomial(p_prev, num_samples=1).squeeze(-1)
        return F.one_hot(samples, num_classes=K).float()

    def compute_kl_divergence(
        self,
        x_t_onehot: torch.Tensor,
        x_0_onehot: torch.Tensor,
        pred_x_0_logits: torch.Tensor,
        t: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Compute KL divergence D_KL(q(x_{t-1} | x_t, x_0) || p_theta(x_{t-1} | x_t))."""
        # True posterior distribution
        q_post = self.q_posterior_probs(x_t_onehot=x_t_onehot, x_0_onehot=x_0_onehot, t=t, K=K)
        # Model predicted distribution
        p_pred = self.p_probs(x_t_onehot=x_t_onehot, pred_x_0_logits=pred_x_0_logits, t=t, K=K)

        kl = q_post * (torch.log(torch.clamp(q_post, min=1e-12)) - torch.log(torch.clamp(p_pred, min=1e-12)))
        kl_per_sample = kl.sum(dim=-1)

        # For t == 0, evaluate exact NLL: -log p_theta(x_0 | x_1)
        pred_log_probs = F.log_softmax(pred_x_0_logits, dim=-1)
        nll_at_zero = -(x_0_onehot * pred_log_probs).sum(dim=-1)

        loss = torch.where(t == 0, nll_at_zero, kl_per_sample)
        return loss.mean()
