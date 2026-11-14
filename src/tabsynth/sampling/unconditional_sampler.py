"""Unconditional reverse diffusion sampler for unlabeled tabular synthesis."""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn

from tabsynth.sampling.mixed_sampler import MixedAncestralSampler


class UnconditionalSampler:
    """Samples synthetic tabular records without class conditioning."""

    def __init__(self, sampler: MixedAncestralSampler):
        self.sampler = sampler

    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int,
        clip_denoised: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Samples a single unconditional batch of mixed-type features."""
        x_num, x_cat_list = self.sampler.sample_step(
            batch_size=batch_size,
            c=None,
            cfg_scale=0.0,
            clip_denoised=clip_denoised,
        )
        return x_num, x_cat_list
