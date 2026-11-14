"""Feature tokenizers, column identity embeddings, and feature type embeddings for tabular data."""

from typing import List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NumericalFeatureTokenizer(nn.Module):
    """Projects scalar continuous tabular features into d_model dimensional token vectors.
    
    For each continuous column i:
        token_i = x_i * w_i + b_i
    where w_i, b_i in R^{d_model}.
    """

    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        assert num_features >= 1, f"num_features must be >= 1, got {num_features}"
        assert d_model >= 2, f"d_model must be >= 2, got {d_model}"
        self.num_features = num_features
        self.d_model = d_model

        # Dedicated weight vector and bias per column: (num_features, d_model)
        self.weights = nn.Parameter(torch.empty(num_features, d_model))
        self.biases = nn.Parameter(torch.empty(num_features, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in = 1
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.biases, -bound, bound)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        """Args:
            x_num: Continuous tensor of shape (batch_size, num_features).
            
        Returns:
            tokens: (batch_size, num_features, d_model).
        """
        assert x_num.ndim == 2, f"x_num must be 2D (B, D), got {x_num.shape}"
        assert x_num.shape[1] == self.num_features, (
            f"Expected {self.num_features} continuous features, got {x_num.shape[1]}"
        )
        # (B, N, 1) * (1, N, d_model) + (1, N, d_model) -> (B, N, d_model)
        tokens = x_num.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)
        return tokens


class CategoricalFeatureTokenizer(nn.Module):
    """Projects discrete categorical features into d_model dimensional token vectors.
    
    Supports both one-hot tensors of shape (batch_size, K_j) and integer indices (batch_size,).
    """

    def __init__(self, cardinalities: List[int], d_model: int):
        super().__init__()
        assert len(cardinalities) >= 1, "cardinalities list must not be empty"
        assert all(k >= 2 for k in cardinalities), f"Each cardinality must be >= 2, got {cardinalities}"
        self.cardinalities = list(cardinalities)
        self.num_features = len(cardinalities)
        self.d_model = d_model

        # Dedicated linear projection per categorical feature
        self.projections = nn.ModuleList([
            nn.Linear(K, d_model, bias=True) for K in cardinalities
        ])

    def forward(self, x_cat_list: List[torch.Tensor]) -> torch.Tensor:
        """Args:
            x_cat_list: List of tensors, one per column.
                        Each tensor is either (batch_size, K_j) one-hot or (batch_size,) integer indices.
                        
        Returns:
            tokens: (batch_size, num_features, d_model).
        """
        assert len(x_cat_list) == self.num_features, (
            f"Expected {self.num_features} categorical columns, got {len(x_cat_list)}"
        )
        tokens = []
        for j, (x_col, proj, K) in enumerate(zip(x_cat_list, self.projections, self.cardinalities)):
            if x_col.ndim == 1 or (x_col.ndim == 2 and x_col.shape[-1] == 1) or x_col.ndim == 0:
                # Integer indices: convert to one-hot on the fly ensuring 2D (B, K)
                indices = x_col.view(-1).long()
                x_oh = F.one_hot(indices, num_classes=K).to(dtype=torch.float32, device=x_col.device)
            else:
                x_oh = x_col.float()
                assert x_oh.shape[-1] == K, f"Expected last dim {K} for column {j}, got {x_oh.shape[-1]}"

            col_token = proj(x_oh)  # (B, d_model)
            tokens.append(col_token)

        # Stack into sequence of feature tokens: (B, num_features, d_model)
        return torch.stack(tokens, dim=1)


class FeatureTokenizer(nn.Module):
    """Unified master tokenizer for mixed-type tabular datasets.
    
    Combines numerical and categorical tokenizers and adds:
    1. Learned Column Identity Embeddings e_col in R^{D x d_model}.
    2. Learned Feature Type Embeddings e_type in R^{2 x d_model} (0 = continuous, 1 = discrete).
    
    Output sequence:
        tokens: (batch_size, num_features, d_model)
    """

    def __init__(
        self,
        num_continuous: int,
        cardinalities: Optional[List[int]] = None,
        d_model: int = 64,
    ):
        super().__init__()
        self.num_continuous = max(0, int(num_continuous))
        self.cardinalities = list(cardinalities or [])
        self.num_categoricals = len(self.cardinalities)
        self.num_features = self.num_continuous + self.num_categoricals
        self.d_model = d_model

        assert self.num_features > 0, "Total number of features (continuous + categorical) must be >= 1"

        # Continuous Tokenizer
        if self.num_continuous > 0:
            self.num_tokenizer = NumericalFeatureTokenizer(num_features=self.num_continuous, d_model=d_model)
        else:
            self.num_tokenizer = None

        # Categorical Tokenizer
        if self.num_categoricals > 0:
            self.cat_tokenizer = CategoricalFeatureTokenizer(cardinalities=self.cardinalities, d_model=d_model)
        else:
            self.cat_tokenizer = None

        # Feature Identity Embeddings: one learned vector per tabular column
        self.col_identity_embeddings = nn.Parameter(torch.empty(self.num_features, d_model))
        nn.init.normal_(self.col_identity_embeddings, std=0.02)

        # Feature Type Embeddings: 0 = continuous, 1 = discrete
        self.type_embeddings = nn.Embedding(2, d_model)
        nn.init.normal_(self.type_embeddings.weight, std=0.02)

        # Precompute type indices buffer: [0]*D_num + [1]*C
        type_indices = [0] * self.num_continuous + [1] * self.num_categoricals
        self.register_buffer("type_indices", torch.tensor(type_indices, dtype=torch.long))

    def forward(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Args:
            x_num: Continuous tensor (batch_size, num_continuous) or None.
            x_cat_list: List of categorical tensors [(batch_size, K_j)] or None.
            
        Returns:
            tokens: (batch_size, num_features, d_model).
        """
        token_list = []

        if self.num_continuous > 0:
            assert x_num is not None, "Continuous features expected but x_num is None"
            token_list.append(self.num_tokenizer(x_num))

        if self.num_categoricals > 0:
            assert x_cat_list is not None, "Categorical features expected but x_cat_list is None"
            token_list.append(self.cat_tokenizer(x_cat_list))

        if len(token_list) == 1:
            tokens = token_list[0]
        else:
            tokens = torch.cat(token_list, dim=1)  # (batch_size, num_features, d_model)

        batch_size = tokens.shape[0]

        # Add Column Identity Embeddings: (1, num_features, d_model)
        tokens = tokens + self.col_identity_embeddings.unsqueeze(0)

        # Add Feature Type Embeddings: (1, num_features, d_model)
        type_emb = self.type_embeddings(self.type_indices).unsqueeze(0)
        tokens = tokens + type_emb

        return tokens
