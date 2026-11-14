"""Specialized output prediction heads for continuous noise and discrete clean-category logits."""

from typing import List, Optional
import torch
import torch.nn as nn


class NumericalOutputHead(nn.Module):
    """Predicts continuous Gaussian noise epsilon for each numerical feature.
    
    Transforms numerical tokens of shape (batch_size, num_continuous, d_model)
    into scalar noise predictions of shape (batch_size, num_continuous).
    """

    def __init__(self, num_continuous: int, d_model: int):
        super().__init__()
        assert num_continuous >= 1, f"num_continuous must be >= 1, got {num_continuous}"
        self.num_continuous = num_continuous
        self.d_model = d_model

        # Dedicated linear projection head per continuous feature
        self.proj = nn.Linear(d_model, 1)

    def forward(self, num_tokens: torch.Tensor) -> torch.Tensor:
        """Args:
            num_tokens: Continuous tokens of shape (batch_size, num_continuous, d_model).
            
        Returns:
            predicted_noise: (batch_size, num_continuous).
        """
        assert num_tokens.ndim == 3, f"Expected 3D tensor (B, D, d_model), got {num_tokens.shape}"
        assert num_tokens.shape[1] == self.num_continuous, (
            f"Expected {self.num_continuous} tokens, got {num_tokens.shape[1]}"
        )
        # (B, num_continuous, d_model) -> (B, num_continuous, 1) -> (B, num_continuous)
        pred_noise = self.proj(num_tokens).squeeze(-1)
        return pred_noise


class CategoricalOutputHeads(nn.Module):
    """Predicts unnormalized clean-state logits for each categorical feature.
    
    Transforms categorical tokens of shape (batch_size, num_categoricals, d_model)
    into a list of unnormalized logit tensors [(batch_size, K_j)].
    """

    def __init__(self, cardinalities: List[int], d_model: int):
        super().__init__()
        assert len(cardinalities) >= 1, "cardinalities list must not be empty"
        self.cardinalities = list(cardinalities)
        self.num_categoricals = len(cardinalities)
        self.d_model = d_model

        # Dedicated logit projection head per categorical feature
        self.heads = nn.ModuleList([
            nn.Linear(d_model, K) for K in cardinalities
        ])

    def forward(self, cat_tokens: torch.Tensor) -> List[torch.Tensor]:
        """Args:
            cat_tokens: Categorical tokens of shape (batch_size, num_categoricals, d_model).
            
        Returns:
            categorical_logits: List of unnormalized logits [(batch_size, K_j)].
        """
        assert cat_tokens.ndim == 3, f"Expected 3D tensor (B, C, d_model), got {cat_tokens.shape}"
        assert cat_tokens.shape[1] == self.num_categoricals, (
            f"Expected {self.num_categoricals} tokens, got {cat_tokens.shape[1]}"
        )
        logits_list = []
        for j, head in enumerate(self.heads):
            token_j = cat_tokens[:, j, :]  # (B, d_model)
            logits_j = head(token_j)        # (B, K_j)
            logits_list.append(logits_j)

        return logits_list
