"""High-level synthetic tabular data generation pipeline with inverse preprocessing."""

from typing import Optional, Dict, List, Tuple, Union
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
from tabsynth.sampling.unconditional_sampler import UnconditionalSampler


class GenerationPipeline:
    """Production tabular synthesis pipeline translating reverse diffusion samples into DataFrames."""

    def __init__(
        self,
        model: nn.Module,
        preprocessor: TabularPreprocessor,
        gaussian_diffusion: Optional[GaussianDiffusion] = None,
        multinomial_diffusion: Optional[CategoricalDiffusion] = None,
        mixed_diffusion: Optional[MixedTypeDiffusion] = None,
        num_timesteps: int = 100,
        device: Union[str, torch.device] = "auto",
    ):
        self.device = select_device(device)
        self.model = model.to(self.device)
        self.preprocessor = preprocessor

        num_continuous = len(preprocessor.schema.continuous_columns)
        cardinalities = preprocessor.schema.cardinalities
        num_classes = len(preprocessor.schema.class_distribution) if preprocessor.schema.class_distribution else 2

        self.mixed_sampler = MixedAncestralSampler(
            model=self.model,
            num_continuous=num_continuous,
            cardinalities=cardinalities,
            gaussian_diffusion=gaussian_diffusion,
            multinomial_diffusion=multinomial_diffusion,
            mixed_diffusion=mixed_diffusion,
            num_timesteps=num_timesteps,
            device=self.device,
        )

        self.conditional_sampler = ConditionalSampler(
            sampler=self.mixed_sampler,
            num_classes=num_classes,
            class_distribution=preprocessor.schema.class_distribution,
            idx_to_target=getattr(preprocessor, "idx_to_target", {}),
        )

        self.unconditional_sampler = UnconditionalSampler(
            sampler=self.mixed_sampler,
        )

    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1000,
        batch_size: int = 256,
        condition_strategy: str = "balanced",
        specific_class: Optional[int] = None,
        cfg_scale: float = 1.0,
        clip_denoised: bool = True,
    ) -> pd.DataFrame:
        """Generates synthetic tabular records in mini-batches and inverse-transforms to original schema.
        
        Args:
            num_samples: Total number of rows to generate.
            batch_size: Mini-batch generation chunk size to bound memory consumption.
            condition_strategy: 'balanced', 'natural', 'class_specific', or 'unconditional'.
            specific_class: Specific class integer if condition_strategy == 'class_specific'.
            cfg_scale: Classifier-Free Guidance scale w >= 0.
            clip_denoised: Clamping continuous features to standard empirical bounds.
            
        Returns:
            pd.DataFrame matching original feature names, data types, and category labels.
        """
        target_col = self.preprocessor.schema.target_column
        is_conditional = target_col is not None and condition_strategy != "unconditional"

        all_x_num: List[np.ndarray] = []
        all_x_cat_lists: List[List[np.ndarray]] = [[] for _ in self.preprocessor.schema.cardinalities]
        all_labels: List[np.ndarray] = []

        samples_remaining = num_samples
        while samples_remaining > 0:
            current_b = min(samples_remaining, batch_size)

            if is_conditional:
                x_num_t, x_cat_t_list, c_tensor = self.conditional_sampler.sample_batch(
                    batch_size=current_b,
                    condition_strategy=condition_strategy,
                    specific_class=specific_class,
                    cfg_scale=cfg_scale,
                    clip_denoised=clip_denoised,
                )
                labels_np = c_tensor.cpu().numpy()
            else:
                x_num_t, x_cat_t_list = self.unconditional_sampler.sample_batch(
                    batch_size=current_b,
                    clip_denoised=clip_denoised,
                )
                labels_np = None

            x_num_np = x_num_t.cpu().numpy()
            if x_num_np.shape[1] > 0:
                all_x_num.append(x_num_np)

            for j in range(len(self.preprocessor.schema.cardinalities)):
                all_x_cat_lists[j].append(x_cat_t_list[j].cpu().numpy())

            if labels_np is not None:
                all_labels.append(labels_np)

            samples_remaining -= current_b

            # Flush MPS cache after large chunks
            if self.device.type == "mps":
                torch.mps.empty_cache()

        # Stack batch chunks
        final_x_num = np.vstack(all_x_num) if len(all_x_num) > 0 else np.empty((num_samples, 0), dtype=np.float32)
        final_x_cat = [np.vstack(cat_chunks) for cat_chunks in all_x_cat_lists]
        final_labels = np.concatenate(all_labels) if len(all_labels) > 0 else None

        # Reconstruct DataFrame via fitted preprocessor
        return self.preprocessor.inverse_transform(
            x_num=final_x_num,
            x_cat_list=final_x_cat,
            labels=final_labels,
        )
