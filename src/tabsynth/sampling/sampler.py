"""Unified class-conditional reverse diffusion sampler facade with Classifier-Free Guidance."""

from typing import Optional, Dict, List, Union, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tabsynth.diffusion.utils import select_device
from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.diffusion.mixed.mixed_diffusion import MixedTypeDiffusion
from tabsynth.data.preprocessor import TabularPreprocessor
from tabsynth.sampling.mixed_sampler import MixedAncestralSampler
from tabsynth.sampling.conditional_sampler import ConditionalSampler
from tabsynth.sampling.generation import GenerationPipeline

# Backward compatibility alias
MultinomialDiffusion = CategoricalDiffusion


class TabSynthSampler:
    """Class-conditional reverse sampler facade for mixed continuous-discrete tabular data."""

    def __init__(
        self,
        model: nn.Module,
        gaussian_diffusion: Optional[GaussianDiffusion] = None,
        multinomial_diffusion: Optional[CategoricalDiffusion] = None,
        preprocessor: Optional[TabularPreprocessor] = None,
        mixed_diffusion: Optional[MixedTypeDiffusion] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = select_device(device) if device is not None else torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.gaussian_diff = gaussian_diffusion
        if self.gaussian_diff is not None and hasattr(self.gaussian_diff, "to_device"):
            self.gaussian_diff.to_device(self.device)

        self.multinomial_diff = multinomial_diffusion
        if self.multinomial_diff is not None and hasattr(self.multinomial_diff, "to_device"):
            self.multinomial_diff.to_device(self.device)

        self.mixed_diff = mixed_diffusion
        if self.mixed_diff is not None and hasattr(self.mixed_diff, "to_device"):
            self.mixed_diff.to_device(self.device)

        self.preprocessor = preprocessor

        num_continuous = len(preprocessor.schema.continuous_columns) if preprocessor else 0
        cardinalities = preprocessor.schema.cardinalities if preprocessor else []
        num_timesteps = (
            gaussian_diffusion.num_timesteps
            if gaussian_diffusion is not None
            else (mixed_diffusion.numerical_timesteps if mixed_diffusion is not None else 100)
        )

        self.ancestral_sampler = MixedAncestralSampler(
            model=self.model,
            num_continuous=num_continuous,
            cardinalities=cardinalities,
            gaussian_diffusion=self.gaussian_diff,
            multinomial_diffusion=self.multinomial_diff,
            mixed_diffusion=self.mixed_diff,
            num_timesteps=num_timesteps,
            device=self.device,
        )

        if self.preprocessor is not None:
            self.pipeline = GenerationPipeline(
                model=self.model,
                preprocessor=self.preprocessor,
                gaussian_diffusion=self.gaussian_diff,
                multinomial_diffusion=self.multinomial_diff,
                mixed_diffusion=self.mixed_diff,
                num_timesteps=num_timesteps,
                device=self.device,
            )
        else:
            self.pipeline = None

    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int,
        c: Optional[torch.Tensor] = None,
        cfg_scale: float = 0.0,
    ) -> Tuple[np.ndarray, List[np.ndarray], Optional[np.ndarray]]:
        """Samples a single batch of mixed-type synthetic records through reverse diffusion."""
        x_num_t, x_cat_t_list = self.ancestral_sampler.sample_step(
            batch_size=batch_size,
            c=c,
            cfg_scale=cfg_scale,
            clip_denoised=True,
        )

        x_num_np = x_num_t.cpu().numpy()
        x_cat_np_list = [cat.cpu().numpy() for cat in x_cat_t_list]
        c_np = c.cpu().numpy() if c is not None else None

        return x_num_np, x_cat_np_list, c_np

    def sample(
        self,
        num_samples: int = 1000,
        batch_size: int = 256,
        condition_strategy: str = "balanced",
        specific_class: Optional[int] = None,
        cfg_scale: float = 1.0,
    ) -> pd.DataFrame:
        """Generates synthetic tabular DataFrame across multiple mini-batches."""
        if self.pipeline is None:
            raise RuntimeError("TabularPreprocessor must be provided to TabSynthSampler to reconstruct DataFrames.")

        return self.pipeline.generate(
            num_samples=num_samples,
            batch_size=batch_size,
            condition_strategy=condition_strategy,
            specific_class=specific_class,
            cfg_scale=cfg_scale,
            clip_denoised=True,
        )
