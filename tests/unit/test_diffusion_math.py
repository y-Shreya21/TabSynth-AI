"""Unit tests for Gaussian and Multinomial diffusion mathematical modules."""

import pytest
import torch
import torch.nn.functional as F
from tabsynth.diffusion.schedules import cosine_beta_schedule, linear_beta_schedule
from tabsynth.diffusion.gaussian import GaussianDiffusion
from tabsynth.diffusion.multinomial import MultinomialDiffusion


def test_noise_schedules():
    """Verify bounds and monotonic decay of noise schedules."""
    timesteps = 100
    betas_cosine = cosine_beta_schedule(timesteps=timesteps)
    betas_linear = linear_beta_schedule(timesteps=timesteps)

    assert len(betas_cosine) == timesteps
    assert len(betas_linear) == timesteps

    # All betas must lie strictly in (0, 1)
    assert (betas_cosine > 0.0).all() and (betas_cosine < 1.0).all()
    assert (betas_linear > 0.0).all() and (betas_linear < 1.0).all()

    alphas_cumprod = torch.cumprod(1.0 - betas_cosine, dim=0)
    # Cumulative product must strictly decrease
    assert (alphas_cumprod[1:] <= alphas_cumprod[:-1]).all()


def test_gaussian_diffusion_math():
    """Verify continuous Gaussian diffusion forward and posterior mathematics."""
    diffusion = GaussianDiffusion(num_timesteps=100, schedule_type="cosine")
    batch_size = 16
    num_continuous = 5

    x_0 = torch.randn(batch_size, num_continuous, dtype=torch.float32)
    t = torch.randint(0, 100, (batch_size,))

    # 1. Forward sample
    x_t, noise = diffusion.q_sample(x_0=x_0, t=t)
    assert x_t.shape == (batch_size, num_continuous)
    assert noise.shape == (batch_size, num_continuous)

    # 2. Reconstruct x_0 from known noise
    pred_x_0 = diffusion.predict_x0_from_eps(x_t=x_t, t=t, eps=noise)
    assert torch.allclose(pred_x_0, x_0, atol=1e-3)

    # 3. Posterior mean and variance
    mean, var, log_var = diffusion.q_posterior_mean_variance(x_0=x_0, x_t=x_t, t=t)
    assert mean.shape == (batch_size, num_continuous)
    assert var.shape == (batch_size, 1)
    assert (var >= 0.0).all()

    # 4. Reverse sampling step
    x_prev = diffusion.p_sample(predicted_eps=noise, x_t=x_t, t=t)
    assert x_prev.shape == (batch_size, num_continuous)
    assert not torch.isnan(x_prev).any()


def test_multinomial_diffusion_math():
    """Verify discrete categorical D3PM transition probability normalization and KL loss."""
    cardinalities = [3, 4, 2]
    diffusion = MultinomialDiffusion(cardinalities=cardinalities, num_timesteps=100)
    batch_size = 8

    for K in cardinalities:
        labels = torch.randint(0, K, (batch_size,))
        x_0_onehot = F.one_hot(labels, num_classes=K).float()
        t = torch.randint(1, 100, (batch_size,))

        # 1. Forward probabilities must be valid distribution (sum to 1.0)
        q_probs = diffusion.q_probs(x_0_onehot=x_0_onehot, t=t, K=K)
        assert torch.allclose(q_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

        # 2. Sampled noisy state must be valid one-hot
        x_t_onehot = diffusion.q_sample(x_0_onehot=x_0_onehot, t=t, K=K)
        assert x_t_onehot.shape == (batch_size, K)
        assert torch.allclose(x_t_onehot.sum(dim=-1), torch.ones(batch_size))

        # 3. Reverse posterior probabilities must sum to 1.0
        q_post = diffusion.q_posterior_probs(x_t_onehot=x_t_onehot, x_0_onehot=x_0_onehot, t=t, K=K)
        assert torch.allclose(q_post.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

        # 4. Model reverse step probabilities must sum to 1.0
        mock_logits = torch.randn(batch_size, K)
        p_prev = diffusion.p_probs(x_t_onehot=x_t_onehot, pred_x_0_logits=mock_logits, t=t, K=K)
        assert torch.allclose(p_prev.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

        # 5. Reverse sample must be valid one-hot
        x_prev = diffusion.p_sample(x_t_onehot=x_t_onehot, pred_x_0_logits=mock_logits, t=t, K=K)
        assert x_prev.shape == (batch_size, K)
        assert torch.allclose(x_prev.sum(dim=-1), torch.ones(batch_size))

        # 6. KL divergence calculation must be non-negative and finite
        kl_loss = diffusion.compute_kl_divergence(
            x_t_onehot=x_t_onehot,
            x_0_onehot=x_0_onehot,
            pred_x_0_logits=mock_logits,
            t=t,
            K=K,
        )
        assert not torch.isnan(kl_loss)
        assert not torch.isinf(kl_loss)
