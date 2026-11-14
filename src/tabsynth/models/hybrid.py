"""TabSynth AI Hybrid Transformer-MLP Denoising Diffusion Model."""

from typing import List, Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from tabsynth.models.tokenizer import FeatureTokenizer
from tabsynth.models.embeddings import (
    MixedTimestepEmbedding,
    SinusoidalTimeEmbedding,
    ClassConditioningEmbedding,
)
from tabsynth.models.transformer import TransformerFeatureEncoder
from tabsynth.models.denoiser import FeatureResidualMLPDenoiser
from tabsynth.models.heads import NumericalOutputHead, CategoricalOutputHeads


class ModelOutput(dict):
    """Container for model outputs supporting dictionary access, attribute access, and tuple unpacking."""

    def __init__(self, predicted_noise: torch.Tensor, categorical_logits: List[torch.Tensor]):
        super().__init__()
        self["predicted_noise"] = predicted_noise
        self["categorical_logits"] = categorical_logits

    @property
    def predicted_noise(self) -> torch.Tensor:
        return self["predicted_noise"]

    @property
    def categorical_logits(self) -> List[torch.Tensor]:
        return self["categorical_logits"]

    def __iter__(self):
        # Enables tuple unpacking: pred_eps, pred_cat_logits = model(...)
        yield self["predicted_noise"]
        yield self["categorical_logits"]


class TabSynthHybridDenoiser(nn.Module):
    """Proposed Hybrid Transformer-MLP Denoising Neural Network for mixed-type tabular diffusion.
    
    Architecture pipeline:
    1. FeatureTokenizer: Projects continuous features and discrete categorical states into d_model tokens,
       augmented with learned column identity and feature type embeddings.
    2. TransformerFeatureEncoder: Models cross-column interdependencies via Multi-Head Self-Attention.
    3. MixedTimestepEmbedding + ClassConditioningEmbedding: Fuses continuous/discrete timesteps and class labels.
    4. FeatureResidualMLPDenoiser: Refines per-feature token sequences conditioned on the fused context.
    5. Specialized Output Heads: Separately projects continuous tokens to Gaussian noise predictions (B, D_num)
       and discrete tokens to clean-category unnormalized logits [(B, K_j)].
    """

    def __init__(
        self,
        num_continuous: int,
        cardinalities: Optional[List[int]] = None,
        num_classes: int = 2,
        d_model: int = 64,
        transformer_layers: int = 2,
        attention_heads: int = 4,
        transformer_ffn_dim: int = 256,
        mlp_hidden_dim: int = 256,
        mlp_layers: int = 2,
        dropout: float = 0.1,
        p_uncond: float = 0.15,
        activation: str = "GELU",
    ):
        super().__init__()
        self.num_continuous = max(0, int(num_continuous))
        self.cardinalities = list(cardinalities or [])
        self.num_categoricals = len(self.cardinalities)
        self.num_features = self.num_continuous + self.num_categoricals
        self.d_model = d_model
        self.num_classes = num_classes

        # 1. Feature Tokenizer (Continuous + Discrete + Column Identity + Feature Type)
        self.tokenizer = FeatureTokenizer(
            num_continuous=self.num_continuous,
            cardinalities=self.cardinalities,
            d_model=d_model,
        )

        # 2. Conditioning Modules
        self.time_embedding = MixedTimestepEmbedding(d_model=d_model)
        self.class_embedding = ClassConditioningEmbedding(
            num_classes=num_classes,
            d_model=d_model,
            p_uncond=p_uncond,
        )

        # 3. Transformer Feature Interaction Backbone
        self.transformer = TransformerFeatureEncoder(
            num_features=self.num_features,
            d_model=d_model,
            nhead=attention_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            activation=activation,
            add_col_embeddings=False,  # Already added in FeatureTokenizer
        )

        # 4. Feature-level Residual MLP Denoising Network
        self.denoiser = FeatureResidualMLPDenoiser(
            d_model=d_model,
            cond_dim=d_model,
            dim_feedforward=mlp_hidden_dim,
            num_layers=mlp_layers,
            dropout=dropout,
            activation=activation,
        )

        # 5. Specialized Output Prediction Heads
        if self.num_continuous > 0:
            self.head_num = NumericalOutputHead(num_continuous=self.num_continuous, d_model=d_model)
        else:
            self.head_num = None

        if self.num_categoricals > 0:
            self.heads_cat = CategoricalOutputHeads(cardinalities=self.cardinalities, d_model=d_model)
        else:
            self.heads_cat = None

    def forward(
        self,
        x_num_t: Optional[torch.Tensor] = None,
        x_cat_t: Optional[List[torch.Tensor]] = None,
        timesteps_num: Optional[torch.Tensor] = None,
        timesteps_cat: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        # Backward compatibility aliases:
        x_cat_t_list: Optional[List[torch.Tensor]] = None,
        t: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        force_uncond: bool = False,
    ) -> ModelOutput:
        """Args:
            x_num_t: Perturbed continuous features (batch_size, num_continuous).
            x_cat_t: List of perturbed categorical tensors [(batch_size, K_j)].
            timesteps_num: 1D tensor of continuous diffusion timesteps (batch_size,).
            timesteps_cat: 1D tensor of discrete diffusion timesteps (batch_size,).
            class_labels: Target class labels (batch_size,) or None.
            condition_mask: Optional boolean tensor for condition dropout.
            
        Returns:
            ModelOutput dict containing:
                "predicted_noise": (batch_size, num_continuous)
                "categorical_logits": List of [(batch_size, K_j)]
        """
        # Resolve aliases
        cat_input = x_cat_t if x_cat_t is not None else x_cat_t_list
        t_num = timesteps_num if timesteps_num is not None else t
        t_cat = timesteps_cat if timesteps_cat is not None else t
        labels = class_labels if class_labels is not None else c

        # Determine batch size
        if x_num_t is not None and x_num_t.ndim >= 2 and x_num_t.shape[0] > 0:
            batch_size = x_num_t.shape[0]
            device = x_num_t.device
        elif cat_input is not None and len(cat_input) > 0:
            batch_size = cat_input[0].shape[0]
            device = cat_input[0].device
        elif t_num is not None:
            batch_size = t_num.shape[0]
            device = t_num.device
        elif t_cat is not None:
            batch_size = t_cat.shape[0]
            device = t_cat.device
        else:
            raise ValueError("Cannot determine batch size from inputs.")

        # 1. Feature Tokenization
        tokens = self.tokenizer(x_num=x_num_t, x_cat_list=cat_input)  # (B, D, d_model)

        # 2. Transformer Contextual Cross-Feature Self-Attention
        context_tokens = self.transformer(tokens)  # (B, D, d_model)

        # 3. Fuse Timestep and Class Conditioning
        e_time = self.time_embedding(t_num=t_num, t_cat=t_cat)  # (B, d_model)
        e_class = self.class_embedding(labels, batch_size=batch_size, force_uncond=force_uncond)  # (B, d_model)
        e_cond = e_time + e_class  # (B, d_model)

        # 4. Feature-level Residual MLP Denoising
        denoised_tokens = self.denoiser(context_tokens, cond=e_cond)  # (B, D, d_model)

        # 5. Output Heads
        # Numerical head operates on the continuous feature tokens
        if self.head_num is not None and self.num_continuous > 0:
            num_tokens = denoised_tokens[:, :self.num_continuous, :]
            pred_eps = self.head_num(num_tokens)
        else:
            pred_eps = torch.empty((batch_size, 0), device=device, dtype=torch.float32)

        # Categorical heads operate on the discrete feature tokens
        if self.heads_cat is not None and self.num_categoricals > 0:
            cat_tokens = denoised_tokens[:, self.num_continuous:, :]
            pred_cat_logits = self.heads_cat(cat_tokens)
        else:
            pred_cat_logits = []

        return ModelOutput(predicted_noise=pred_eps, categorical_logits=pred_cat_logits)


# Backward-compatible alias
TabSynthHybridModel = TabSynthHybridDenoiser
