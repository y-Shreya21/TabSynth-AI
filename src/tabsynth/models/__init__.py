"""TabSynth AI Model Architecture Package."""

from tabsynth.models.tokenizer import (
    NumericalFeatureTokenizer,
    CategoricalFeatureTokenizer,
    FeatureTokenizer,
)
from tabsynth.models.embeddings import (
    SinusoidalTimeEmbedding,
    MixedTimestepEmbedding,
    ClassConditioningEmbedding,
)
from tabsynth.models.transformer import (
    TransformerEncoderBlock,
    TransformerFeatureEncoder,
)
from tabsynth.models.denoiser import (
    FeatureResidualMLPBlock,
    FeatureResidualMLPDenoiser,
    ResidualMLPBlock,
    ResidualMLPDenoiser,
)
from tabsynth.models.heads import (
    NumericalOutputHead,
    CategoricalOutputHeads,
)
from tabsynth.models.hybrid import (
    TabSynthHybridDenoiser,
    TabSynthHybridModel,
    ModelOutput,
)
from tabsynth.models.mlp_baseline import (
    TabSynthMLPDenoiser,
    TabDDPMMLPModel,
)
from tabsynth.models.integration import ModelDiffusionIntegrator
from tabsynth.models.diagnostics import (
    count_parameters,
    get_model_summary,
    compare_models,
)

__all__ = [
    # Tokenizers
    "NumericalFeatureTokenizer",
    "CategoricalFeatureTokenizer",
    "FeatureTokenizer",
    # Embeddings
    "SinusoidalTimeEmbedding",
    "MixedTimestepEmbedding",
    "ClassConditioningEmbedding",
    # Transformer
    "TransformerEncoderBlock",
    "TransformerFeatureEncoder",
    # Denoiser
    "FeatureResidualMLPBlock",
    "FeatureResidualMLPDenoiser",
    "ResidualMLPBlock",
    "ResidualMLPDenoiser",
    # Heads
    "NumericalOutputHead",
    "CategoricalOutputHeads",
    # Models
    "TabSynthHybridDenoiser",
    "TabSynthHybridModel",
    "TabSynthMLPDenoiser",
    "TabDDPMMLPModel",
    "ModelOutput",
    # Integration
    "ModelDiffusionIntegrator",
    # Diagnostics
    "count_parameters",
    "get_model_summary",
    "compare_models",
]
