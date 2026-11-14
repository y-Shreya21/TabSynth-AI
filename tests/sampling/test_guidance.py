"""Unit tests for Classifier-Free Guidance (CFG) extrapolation."""

import pytest
import torch
from tabsynth.sampling.guidance import (
    apply_continuous_cfg,
    apply_categorical_cfg,
    CFGController,
)


def test_continuous_cfg_extrapolation():
    """Verify formula: eps_guided = (1 + w) * eps_cond - w * eps_uncond."""
    b, d = 4, 3
    eps_cond = torch.tensor([[1.0, 2.0, 3.0]] * b)
    eps_uncond = torch.tensor([[0.0, 0.0, 0.0]] * b)

    # When w = 0, guided == cond
    guided_w0 = apply_continuous_cfg(eps_cond, eps_uncond, cfg_scale=0.0)
    assert torch.allclose(guided_w0, eps_cond)

    # When w = 1.0, guided = 2 * cond - uncond = 2 * cond
    guided_w1 = apply_continuous_cfg(eps_cond, eps_uncond, cfg_scale=1.0)
    expected_w1 = 2.0 * eps_cond
    assert torch.allclose(guided_w1, expected_w1)

    # When w = 2.0, guided = 3 * cond - 2 * uncond = 3 * cond
    guided_w2 = apply_continuous_cfg(eps_cond, eps_uncond, cfg_scale=2.0)
    expected_w2 = 3.0 * eps_cond
    assert torch.allclose(guided_w2, expected_w2)


def test_categorical_cfg_extrapolation():
    """Verify discrete logit extrapolation with CFG."""
    logits_cond = [torch.tensor([[2.0, -1.0, 0.5]])]
    logits_uncond = [torch.tensor([[0.0, 0.0, 0.0]])]

    # When w = 1.0, guided = 2 * cond - uncond = 2 * cond
    guided = apply_categorical_cfg(logits_cond, logits_uncond, cfg_scale=1.0)
    assert torch.allclose(guided[0], 2.0 * logits_cond[0])


def test_cfg_controller():
    """Verify CFGController extrapolates both heads correctly."""
    controller = CFGController(cfg_scale=1.5)
    eps_c = torch.tensor([[2.0, 4.0]])
    eps_u = torch.tensor([[1.0, 1.0]])
    cat_c = [torch.tensor([[1.0, 2.0]])]
    cat_u = [torch.tensor([[0.0, 0.0]])]

    guided_eps, guided_cat = controller.extrapolate(eps_c, eps_u, cat_c, cat_u)

    # (1 + 1.5)*2 - 1.5*1 = 5 - 1.5 = 3.5
    # (1 + 1.5)*4 - 1.5*1 = 10 - 1.5 = 8.5
    assert torch.allclose(guided_eps, torch.tensor([[3.5, 8.5]]))
    assert torch.allclose(guided_cat[0], torch.tensor([[2.5, 5.0]]))
