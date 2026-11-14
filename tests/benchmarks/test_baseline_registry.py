"""Tests for Baseline Synthesizer Registry."""

import pytest
from tabsynth.evaluation.benchmarks.baseline_registry import (
    list_available_synthesizers,
    get_synthesizer,
    SYNTHESIZER_REGISTRY,
)
from tabsynth.evaluation.benchmarks.copula_wrapper import GaussianCopulaSynthesizer


def test_registry_listing():
    available = list_available_synthesizers()
    assert "gaussian_copula" in available
    assert "tvae" in available
    assert "tabddpm_mlp" in available
    assert "tabsynth_hybrid" in available


def test_get_synthesizer_success():
    synth = get_synthesizer("gaussian_copula", seed=123)
    assert isinstance(synth, GaussianCopulaSynthesizer)
    assert synth.seed == 123


def test_get_synthesizer_unknown():
    with pytest.raises(KeyError):
        get_synthesizer("non_existent_model")
