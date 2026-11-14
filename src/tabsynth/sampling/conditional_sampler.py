"""Class-conditional sampling coordinator with Classifier-Free Guidance."""

from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from tabsynth.sampling.mixed_sampler import MixedAncestralSampler


class ConditionalSampler:
    """Coordinates class-conditional generation supporting various target distribution policies."""

    def __init__(
        self,
        sampler: MixedAncestralSampler,
        num_classes: int = 2,
        class_distribution: Optional[Dict[str, float]] = None,
        idx_to_target: Optional[Dict[int, str]] = None,
    ):
        self.sampler = sampler
        self.num_classes = max(1, num_classes)
        self.class_distribution = class_distribution
        self.idx_to_target = idx_to_target or {}

    def generate_conditions(
        self,
        batch_size: int,
        condition_strategy: str = "balanced",
        specific_class: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Samples conditioning target class labels according to the specified strategy.
        
        Strategies:
            - 'balanced': Uniform probability across all classes 0..C-1.
            - 'natural': Empirical prior probability distribution matching training dataset.
            - 'class_specific': Constant specified class label for all samples.
        """
        target_device = device or self.sampler.device
        strat = condition_strategy.lower()

        if strat == "class_specific":
            target_cls = int(specific_class if specific_class is not None else 0)
            target_cls = max(0, min(self.num_classes - 1, target_cls))
            return torch.full((batch_size,), target_cls, device=target_device, dtype=torch.long)

        elif strat == "natural" and self.class_distribution is not None and len(self.class_distribution) > 0:
            # Empirical distribution sampling
            p_values = []
            for i in range(self.num_classes):
                str_label = self.idx_to_target.get(i, str(i))
                p_values.append(self.class_distribution.get(str_label, 1.0 / self.num_classes))
            p_arr = np.array(p_values, dtype=np.float64)
            p_sum = p_arr.sum()
            p_norm = p_arr / (p_sum if p_sum > 0 else 1.0)

            sampled_indices = np.random.choice(self.num_classes, size=batch_size, p=p_norm)
            return torch.tensor(sampled_indices, device=target_device, dtype=torch.long)

        else:
            # Balanced uniform strategy: 0..C-1
            return torch.randint(0, self.num_classes, (batch_size,), device=target_device, dtype=torch.long)

    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int,
        condition_strategy: str = "balanced",
        specific_class: Optional[int] = None,
        cfg_scale: float = 1.0,
        clip_denoised: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Samples a single conditional batch of mixed-type synthetic features and target labels."""
        c = self.generate_conditions(
            batch_size=batch_size,
            condition_strategy=condition_strategy,
            specific_class=specific_class,
        )

        x_num, x_cat_list = self.sampler.sample_step(
            batch_size=batch_size,
            c=c,
            cfg_scale=cfg_scale,
            clip_denoised=clip_denoised,
        )

        return x_num, x_cat_list, c
