"""Diffusion model wrappers adhering to the BaseSynthesizer interface."""

import os
import time
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tabsynth.data.schema import TableSchema, infer_table_schema
from tabsynth.data.preprocessor import TabularPreprocessor
from tabsynth.diffusion.utils import select_device
from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.models.hybrid import TabSynthHybridDenoiser
from tabsynth.models.mlp_baseline import TabSynthMLPDenoiser
from tabsynth.training.checkpoint import CheckpointManager
from tabsynth.sampling.generation import GenerationPipeline
from .base import BaseSynthesizer


class TabSynthDiffusionBaseSynthesizer(BaseSynthesizer):
    """Unified wrapper for TabSynth diffusion models (Hybrid and MLP)."""

    def __init__(
        self,
        name: str,
        model_type: str = "hybrid",
        checkpoint_path: Optional[str] = None,
        d_model: int = 64,
        num_timesteps: int = 100,
        device: str = "auto",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(name=name, model_type=model_type, **kwargs)
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.d_model = d_model
        self.num_timesteps = num_timesteps
        self.device = select_device(device)
        self.seed = seed

        self.model: Optional[nn.Module] = None
        self.preprocessor: Optional[TabularPreprocessor] = None
        self.pipeline: Optional[GenerationPipeline] = None
        self.schema: Optional[TableSchema] = None
        self.target_column: Optional[str] = None

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self.load(checkpoint_path)

    def _build_model(
        self,
        num_continuous: int,
        cardinalities: List[int],
        num_classes: int,
    ) -> nn.Module:
        """Constructs the appropriate neural network architecture."""
        if self.model_type in ("hybrid", "hybrid_transformer_mlp"):
            return TabSynthHybridDenoiser(
                num_continuous=num_continuous,
                cardinalities=cardinalities,
                num_classes=num_classes,
                d_model=self.d_model,
                transformer_layers=2,
                attention_heads=4,
                transformer_ffn_dim=256,
                mlp_hidden_dim=256,
                mlp_layers=2,
                p_uncond=0.15,
            )
        else:
            return TabSynthMLPDenoiser(
                num_continuous=num_continuous,
                cardinalities=cardinalities,
                num_classes=num_classes,
                d_model=self.d_model,
                mlp_hidden_dim=256,
                mlp_layers=3,
                p_uncond=0.15,
            )

    def fit(self, df_train: pd.DataFrame, target_column: Optional[str] = None) -> "TabSynthDiffusionBaseSynthesizer":
        """Fits preprocessor and diffusion model on training data."""
        start_t = time.time()
        self.target_column = target_column
        self.schema = infer_table_schema(df_train, target_column=target_column)

        self.preprocessor = TabularPreprocessor(schema=self.schema, seed=self.seed)
        self.preprocessor.fit(df_train)

        num_cont = len(self.schema.continuous_columns)
        cards = self.schema.cardinalities
        num_classes = len(self.schema.class_distribution) if self.schema.class_distribution else 2

        self.model = self._build_model(num_cont, cards, num_classes).to(self.device)
        self.parameter_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        gaussian_diff = GaussianDiffusion(num_timesteps=self.num_timesteps, schedule_type="cosine", device=self.device)
        categorical_diff = CategoricalDiffusion(cardinalities=cards, num_timesteps=self.num_timesteps, schedule_type="cosine", device=self.device)

        self.pipeline = GenerationPipeline(
            model=self.model,
            preprocessor=self.preprocessor,
            gaussian_diffusion=gaussian_diff,
            multinomial_diffusion=categorical_diff,
            num_timesteps=self.num_timesteps,
            device=self.device,
        )

        self.training_time_sec = float(time.time() - start_t)
        self.is_fitted = True
        return self

    def sample(self, num_samples: int = 1000, batch_size: int = 256) -> pd.DataFrame:
        """Samples synthetic records."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError(f"{self.name} must be fitted or loaded before sampling.")

        strat = "natural" if self.schema and self.schema.target_column else "unconditional"
        return self.pipeline.generate(
            num_samples=num_samples,
            batch_size=batch_size,
            condition_strategy=strat,
            cfg_scale=1.0,
            clip_denoised=True,
        )

    def sample_conditional(self, class_condition: Any, num_samples: int = 1000, batch_size: int = 256) -> pd.DataFrame:
        """Samples class-conditioned synthetic records."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError(f"{self.name} must be fitted or loaded before sampling.")

        # Map class label to class index if needed
        class_idx = 0
        if self.preprocessor is not None and hasattr(self.preprocessor, "target_to_idx"):
            target_map = getattr(self.preprocessor, "target_to_idx", {})
            class_str = str(class_condition)
            if class_str in target_map:
                class_idx = target_map[class_str]
            elif isinstance(class_condition, int):
                class_idx = class_condition

        return self.pipeline.generate(
            num_samples=num_samples,
            batch_size=batch_size,
            condition_strategy="class_specific",
            specific_class=class_idx,
            cfg_scale=1.5,
            clip_denoised=True,
        )

    def save(self, path: str) -> None:
        """Saves weights and preprocessor."""
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model is not initialized.")
        CheckpointManager.save(
            model=self.model,
            optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-3),
            epoch=1,
            val_loss=0.0,
            preprocessor=self.preprocessor,
            config={"model": {"type": self.model_type, "d_model": self.d_model}, "diffusion": {"steps": self.num_timesteps}},
            filepath=path,
        )

    def load(self, path: str) -> "TabSynthDiffusionBaseSynthesizer":
        """Loads weights and preprocessor from checkpoint."""
        ckpt = CheckpointManager.load(checkpoint_path=path, device=self.device)
        self.preprocessor = ckpt["preprocessor"]
        self.schema = self.preprocessor.schema
        self.target_column = self.schema.target_column

        cfg = ckpt.get("config", {}) or {}
        self.num_timesteps = cfg.get("diffusion", {}).get("steps", self.num_timesteps)
        self.d_model = cfg.get("model", {}).get("d_model", self.d_model)

        num_cont = len(self.schema.continuous_columns)
        cards = self.schema.cardinalities
        num_classes = len(self.schema.class_distribution) if self.schema.class_distribution else 2

        self.model = self._build_model(num_cont, cards, num_classes).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.parameter_count = sum(p.numel() for p in self.model.parameters())

        gaussian_diff = GaussianDiffusion(num_timesteps=self.num_timesteps, schedule_type="cosine", device=self.device)
        categorical_diff = CategoricalDiffusion(cardinalities=cards, num_timesteps=self.num_timesteps, schedule_type="cosine", device=self.device)

        self.pipeline = GenerationPipeline(
            model=self.model,
            preprocessor=self.preprocessor,
            gaussian_diffusion=gaussian_diff,
            multinomial_diffusion=categorical_diff,
            num_timesteps=self.num_timesteps,
            device=self.device,
        )
        self.is_fitted = True
        return self


class TabSynthHybridSynthesizer(TabSynthDiffusionBaseSynthesizer):
    """TabSynth Hybrid Transformer-MLP diffusion model synthesizer."""

    def __init__(self, checkpoint_path: Optional[str] = None, **kwargs):
        super().__init__(name="tabsynth_hybrid", model_type="hybrid", checkpoint_path=checkpoint_path, **kwargs)


class TabDDPMMLPSynthesizer(TabSynthDiffusionBaseSynthesizer):
    """TabDDPM MLP diffusion model synthesizer baseline."""

    def __init__(self, checkpoint_path: Optional[str] = None, **kwargs):
        super().__init__(name="tabddpm_mlp", model_type="mlp", checkpoint_path=checkpoint_path, **kwargs)
