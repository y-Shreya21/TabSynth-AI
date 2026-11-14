"""Unit tests for sinusoidal and mixed timestep embeddings."""

import pytest
import torch

from tabsynth.models.embeddings import SinusoidalTimeEmbedding, MixedTimestepEmbedding


def test_sinusoidal_time_embedding_shapes_and_values():
    B, d_model = 16, 64
    time_emb = SinusoidalTimeEmbedding(d_model=d_model)

    t = torch.randint(0, 100, (B,))
    emb = time_emb(t)

    assert emb.shape == (B, d_model)
    assert not torch.isnan(emb).any()
    assert not torch.isinf(emb).any()

    # Different timesteps should have different embeddings
    t_diff = torch.tensor([0, 10, 50, 99])
    emb_diff = time_emb(t_diff)
    for i in range(len(t_diff)):
        for j in range(i + 1, len(t_diff)):
            assert not torch.allclose(emb_diff[i], emb_diff[j], atol=1e-3)


def test_mixed_timestep_embedding_dual_and_single_inputs():
    B, d_model = 8, 64
    mixed_emb = MixedTimestepEmbedding(d_model=d_model)

    t_num = torch.randint(0, 100, (B,))
    t_cat = torch.randint(0, 100, (B,))

    # 1. Both timesteps provided
    fused_emb = mixed_emb(t_num=t_num, t_cat=t_cat)
    assert fused_emb.shape == (B, d_model)
    assert not torch.isnan(fused_emb).any()

    # 2. Only numerical timestep provided
    num_only_emb = mixed_emb(t_num=t_num, t_cat=None)
    assert num_only_emb.shape == (B, d_model)

    # 3. Only categorical timestep provided
    cat_only_emb = mixed_emb(t_num=None, t_cat=t_cat)
    assert cat_only_emb.shape == (B, d_model)

    # 4. Neither provided raises AssertionError
    with pytest.raises(AssertionError):
        mixed_emb(t_num=None, t_cat=None)
