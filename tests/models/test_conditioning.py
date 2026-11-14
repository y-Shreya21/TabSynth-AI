"""Unit tests for class conditioning and classifier-free guidance dropout."""

import pytest
import torch

from tabsynth.models.embeddings import ClassConditioningEmbedding


def test_class_conditioning_embedding_shapes():
    num_classes, d_model = 3, 64
    class_emb = ClassConditioningEmbedding(num_classes=num_classes, d_model=d_model, p_uncond=0.0)

    # Class indices 0, 1, 2
    c = torch.tensor([0, 1, 2, 0])
    emb = class_emb(c, batch_size=len(c))
    assert emb.shape == (len(c), d_model)
    assert not torch.isnan(emb).any()


def test_unconditional_null_token():
    num_classes, d_model = 2, 32
    class_emb = ClassConditioningEmbedding(num_classes=num_classes, d_model=d_model, p_uncond=0.0)

    B = 5
    # 1. When c is None, should produce identical embeddings for all rows (null token)
    emb_none = class_emb(None, batch_size=B)
    assert emb_none.shape == (B, d_model)
    for i in range(1, B):
        assert torch.allclose(emb_none[0], emb_none[i])

    # 2. When force_uncond is True, should match emb_none
    c = torch.tensor([0, 1, 0, 1, 0])
    emb_forced = class_emb(c, batch_size=B, force_uncond=True)
    assert torch.allclose(emb_none, emb_forced)


def test_conditioning_dropout_during_training():
    num_classes, d_model = 2, 32
    # Set high dropout probability to guarantee drops in a batch
    class_emb = ClassConditioningEmbedding(num_classes=num_classes, d_model=d_model, p_uncond=0.5)

    B = 100
    c = torch.zeros(B, dtype=torch.long)  # All class 0
    null_emb = class_emb(None, batch_size=1)[0]

    # In training mode with p_uncond=0.5: some tokens will be dropped to null
    class_emb.train()
    emb_train = class_emb(c, batch_size=B)
    dropped_count = sum(torch.allclose(emb_train[i], null_emb) for i in range(B))
    assert 20 < dropped_count < 80, f"Expected ~50 dropped tokens, got {dropped_count}"

    # In eval mode: no tokens should be dropped
    class_emb.eval()
    emb_eval = class_emb(c, batch_size=B)
    eval_dropped_count = sum(torch.allclose(emb_eval[i], null_emb) for i in range(B))
    assert eval_dropped_count == 0


def test_class_embedding_gradient_flow():
    class_emb = ClassConditioningEmbedding(num_classes=2, d_model=16, p_uncond=0.0)
    c = torch.tensor([0, 1])
    emb = class_emb(c, batch_size=2)
    loss = emb.sum()
    loss.backward()

    assert class_emb.embedding.weight.grad is not None
    assert torch.norm(class_emb.embedding.weight.grad) > 0.0
