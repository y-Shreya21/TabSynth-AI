"""TabDDPM-style MLP-only diffusion baseline model for architectural ablation."""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from tabsynth.models.embeddings import SinusoidalTimeEmbedding, ClassConditioningEmbedding
from tabsynth.models.denoiser import ResidualMLPDenoiser


class TabDDPMMLPModel(nn.Module):
    """MLP-only diffusion baseline model corresponding to the TabDDPM architecture.
    
    Flattens continuous and one-hot categorical inputs directly into a multi-layer
    residual MLP denoiser without Transformer feature-attention tokenization.
    Used to isolate the exact empirical value of cross-feature self-attention (RQ1 / E1).
    """

    def __init__(
        self,
        num_continuous: int,
        cardinalities: List[int],
        num_classes: int = 2,
        d_model: int = 64,
        mlp_hidden_dim: int = 256,
        mlp_layers: int = 3,
        dropout: float = 0.1,
        p_uncond: float = 0.15,
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.cardinalities = cardinalities
        self.total_cat_dim = sum(cardinalities)
        self.raw_input_dim = num_continuous + self.total_cat_dim
        self.d_model = d_model

        # Conditioning modules
        self.time_embedding = SinusoidalTimeEmbedding(d_model=d_model)
        self.class_embedding = ClassConditioningEmbedding(
            num_classes=num_classes,
            d_model=d_model,
            p_uncond=p_uncond,
        )

        # Residual MLP Denoising Network
        self.denoiser = ResidualMLPDenoiser(
            in_dim=self.raw_input_dim,
            cond_dim=d_model,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_layers,
            num_continuous=num_continuous,
            cardinalities=cardinalities,
            dropout=dropout,
        )

    def forward(
        self,
        x_num_t: Optional[torch.Tensor],
        x_cat_t_list: Optional[List[torch.Tensor]],
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        force_uncond: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Directly concatenates raw inputs into the MLP denoiser."""
        batch_size = t.shape[0]

        # Concatenate continuous features and one-hot categorical vectors
        inputs = []
        if self.num_continuous > 0 and x_num_t is not None:
            inputs.append(x_num_t)
        if len(self.cardinalities) > 0 and x_cat_t_list is not None:
            inputs.extend(x_cat_t_list)

        raw_flat = torch.cat(inputs, dim=-1)

        # Fuse conditioning
        time_emb = self.time_embedding(t)
        class_emb = self.class_embedding(c, batch_size=batch_size, force_uncond=force_uncond)
        e_cond = time_emb + class_emb

        pred_eps, pred_cat_logits = self.denoiser(h_flat=raw_flat, cond=e_cond)
        return pred_eps, pred_cat_logits
