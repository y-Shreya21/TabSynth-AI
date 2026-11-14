"""Abstract BaseSynthesizer interface for all generative baselines."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd


class BaseSynthesizer(ABC):
    """Abstract interface for tabular synthetic data generators."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs
        self.is_fitted: bool = False
        self.training_time_sec: float = 0.0
        self.parameter_count: int = 0

    @abstractmethod
    def fit(self, df_train: pd.DataFrame, target_column: Optional[str] = None) -> "BaseSynthesizer":
        """Fits the generative model on the real training DataFrame."""
        pass

    @abstractmethod
    def sample(self, num_samples: int) -> pd.DataFrame:
        """Generates unconditional synthetic tabular records."""
        pass

    @abstractmethod
    def sample_conditional(self, class_condition: Any, num_samples: int) -> pd.DataFrame:
        """Generates class-conditioned synthetic tabular records."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Saves model weights and preprocessing parameters."""
        pass

    @abstractmethod
    def load(self, path: str) -> "BaseSynthesizer":
        """Loads model weights and parameters from disk."""
        pass
