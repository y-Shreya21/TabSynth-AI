"""TabSynth AI Tabular Diffusion Engine.

Provides modular numerical (Gaussian), categorical (Multinomial D3PM), and unified mixed-type
diffusion implementations.
"""

from tabsynth.diffusion.schedules import (
    linear_beta_schedule,
    cosine_beta_schedule,
    get_noise_schedule,
    PrecomputedDiffusionConstants,
)
from tabsynth.diffusion.embeddings import SinusoidalTimeEmbedding, TimestepMLP
from tabsynth.diffusion.utils import (
    select_device,
    extract_into_tensor,
    check_probability_simplex,
    to_one_hot,
    from_one_hot,
)
from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.numerical.numerical_sampler import NumericalSampler
from tabsynth.diffusion.categorical.transition import (
    get_transition_matrix,
    get_cumulative_transition_matrix,
    validate_transition_matrix,
)
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.diffusion.categorical.categorical_sampler import CategoricalSampler
from tabsynth.diffusion.mixed.mixed_diffusion import MixedTypeDiffusion

# Backward-compatibility alias
MultinomialDiffusion = CategoricalDiffusion

__all__ = [
    # Schedules
    "linear_beta_schedule",
    "cosine_beta_schedule",
    "get_noise_schedule",
    "PrecomputedDiffusionConstants",
    # Embeddings
    "SinusoidalTimeEmbedding",
    "TimestepMLP",
    # Utilities
    "select_device",
    "extract_into_tensor",
    "check_probability_simplex",
    "to_one_hot",
    "from_one_hot",
    # Numerical
    "GaussianDiffusion",
    "NumericalSampler",
    # Categorical
    "get_transition_matrix",
    "get_cumulative_transition_matrix",
    "validate_transition_matrix",
    "CategoricalDiffusion",
    "CategoricalSampler",
    "MultinomialDiffusion",
    # Mixed
    "MixedTypeDiffusion",
]
