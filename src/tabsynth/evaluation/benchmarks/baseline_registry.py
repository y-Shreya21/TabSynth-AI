"""Registry and factory for generative tabular baselines."""

from typing import Dict, List, Any, Type, Optional
from .base import BaseSynthesizer
from .copula_wrapper import GaussianCopulaSynthesizer
from .tvae import TVAESynthesizer
from .diffusion_wrappers import (
    TabSynthHybridSynthesizer,
    TabDDPMMLPSynthesizer,
)

SYNTHESIZER_REGISTRY: Dict[str, Type[BaseSynthesizer]] = {
    "gaussian_copula": GaussianCopulaSynthesizer,
    "tvae": TVAESynthesizer,
    "tabddpm_mlp": TabDDPMMLPSynthesizer,
    "tabsynth_mlp": TabDDPMMLPSynthesizer,
    "tabsynth_hybrid": TabSynthHybridSynthesizer,
}


def get_synthesizer(name: str, **kwargs) -> BaseSynthesizer:
    """Instantiates a baseline synthesizer by registered name."""
    clean_name = name.lower().strip()
    if clean_name not in SYNTHESIZER_REGISTRY:
        raise KeyError(
            f"Synthesizer '{name}' not found in registry. "
            f"Available baselines: {list(SYNTHESIZER_REGISTRY.keys())}"
        )
    cls = SYNTHESIZER_REGISTRY[clean_name]
    return cls(**kwargs)


def list_available_synthesizers() -> List[str]:
    """Returns the names of all registered synthesizers."""
    return list(SYNTHESIZER_REGISTRY.keys())
