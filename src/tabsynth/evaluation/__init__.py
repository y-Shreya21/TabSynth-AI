"""TabSynth Multi-Dimensional Evaluation: Statistical Fidelity, Downstream Utility & Privacy."""

from .fidelity import FidelityEvaluator, evaluate_statistical_fidelity
from .utility import UtilityEvaluator, evaluate_downstream_utility
from .privacy import PrivacyAuditor, evaluate_empirical_privacy
from .benchmarks import BenchmarkRunner, get_synthesizer, list_available_synthesizers
from .visualization import (
    plot_marginal_distributions,
    plot_correlation_difference,
    plot_dcr_distributions,
    plot_augmentation_curves,
    save_figure,
)

__all__ = [
    "FidelityEvaluator",
    "evaluate_statistical_fidelity",
    "UtilityEvaluator",
    "evaluate_downstream_utility",
    "PrivacyAuditor",
    "evaluate_empirical_privacy",
    "BenchmarkRunner",
    "get_synthesizer",
    "list_available_synthesizers",
    "plot_marginal_distributions",
    "plot_correlation_difference",
    "plot_dcr_distributions",
    "plot_augmentation_curves",
    "save_figure",
]
