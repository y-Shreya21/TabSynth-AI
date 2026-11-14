"""Research ablation experiment suite for TabSynth AI."""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from tabsynth.data.schema import TableSchema
from tabsynth.evaluation.benchmarks.base import BaseSynthesizer
from tabsynth.evaluation.fidelity.fidelity_report import FidelityEvaluator
from tabsynth.evaluation.utility.utility_report import UtilityEvaluator
from tabsynth.evaluation.privacy.privacy_report import PrivacyAuditor


def run_e1_architecture_ablation(
    hybrid_synth: BaseSynthesizer,
    mlp_synth: BaseSynthesizer,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    schema: TableSchema,
    num_samples: int = 500,
) -> Dict[str, Any]:
    """E1: Isolates the contribution of Transformer cross-feature attention vs pure MLP.
    
    Compares correlation Frobenius distance, C2ST ROC-AUC, and downstream Macro-F1.
    """
    fidelity_eval = FidelityEvaluator(schema.continuous_columns, schema.categorical_columns)
    utility_eval = UtilityEvaluator(target_column=schema.target_column)

    # 1. Sample from both models
    df_hybrid = hybrid_synth.sample(num_samples)
    df_mlp = mlp_synth.sample(num_samples)

    # 2. Evaluate Fidelity
    fid_hybrid = fidelity_eval.evaluate(df_train, df_hybrid)
    fid_mlp = fidelity_eval.evaluate(df_train, df_mlp)

    # 3. Evaluate Utility
    util_hybrid = utility_eval.evaluate(df_train, df_hybrid, df_test)
    util_mlp = utility_eval.evaluate(df_train, df_mlp, df_test)

    return {
        "experiment": "E1_Architecture_Ablation",
        "hybrid": {
            "fidelity": fid_hybrid["summary"],
            "mean_f1_parity": util_hybrid["mean_f1_parity"],
            "parameters": hybrid_synth.parameter_count,
        },
        "mlp": {
            "fidelity": fid_mlp["summary"],
            "mean_f1_parity": util_mlp["mean_f1_parity"],
            "parameters": mlp_synth.parameter_count,
        },
        "correlation_frobenius_gain": float(
            fid_mlp["summary"]["pearson_frobenius_distance"] - fid_hybrid["summary"]["pearson_frobenius_distance"]
        ),
        "f1_parity_gain": float(
            util_hybrid["mean_f1_parity"] - util_mlp["mean_f1_parity"]
        ),
    }


def run_e5_class_sampling_ablation(
    synth: BaseSynthesizer,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    schema: TableSchema,
    num_samples: int = 500,
) -> Dict[str, Any]:
    """E5: Evaluates Balanced (50/50) vs Natural/Empirical class sampling strategies."""
    if schema.target_column is None:
        raise ValueError("Target column required for class sampling ablation.")

    utility_eval = UtilityEvaluator(target_column=schema.target_column)

    # 1. Natural Sampling
    df_natural = synth.sample(num_samples)
    util_natural = utility_eval.evaluate(df_train, df_natural, df_test)

    # 2. Balanced Sampling
    classes = list(schema.class_distribution.keys()) if schema.class_distribution else [0, 1]
    n_per_class = num_samples // len(classes)
    balanced_chunks = []
    for c in classes:
        balanced_chunks.append(synth.sample_conditional(c, n_per_class))
    df_balanced = pd.concat(balanced_chunks, axis=0).reset_index(drop=True)
    util_balanced = utility_eval.evaluate(df_train, df_balanced, df_test)

    rf_nat = util_natural["tstr"]["random_forest"]
    rf_bal = util_balanced["tstr"]["random_forest"]

    return {
        "experiment": "E5_Class_Sampling_Ablation",
        "natural_class_distribution": df_natural[schema.target_column].value_counts(normalize=True).to_dict(),
        "balanced_class_distribution": df_balanced[schema.target_column].value_counts(normalize=True).to_dict(),
        "natural_rf_macro_f1": rf_nat["tstr"]["macro_f1"],
        "balanced_rf_macro_f1": rf_bal["tstr"]["macro_f1"],
        "natural_rf_minority_recall": rf_nat["tstr"]["minority_recall"],
        "balanced_rf_minority_recall": rf_bal["tstr"]["minority_recall"],
        "minority_recall_gain_from_balancing": float(
            rf_bal["tstr"]["minority_recall"] - rf_nat["tstr"]["minority_recall"]
        ),
    }


def run_e6_minority_augmentation_study(
    synth: BaseSynthesizer,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    schema: TableSchema,
    ratios: Optional[List[float]] = None,
    num_samples: int = 1000,
) -> Dict[str, Any]:
    """E6: Evaluates downstream performance across minority augmentation ratios."""
    utility_eval = UtilityEvaluator(
        target_column=schema.target_column,
        augmentation_ratios=ratios or [0.25, 0.5, 1.0, 2.0],
    )
    df_synth = synth.sample(num_samples)
    rep = utility_eval.evaluate(df_train, df_synth, df_test)
    return {
        "experiment": "E6_Minority_Augmentation",
        "augmentation_results": rep["augmentation"],
    }


def run_e7_fidelity_privacy_tradeoff(
    bench_results: Dict[str, Any],
) -> Dict[str, Any]:
    """E7: Analyzes fidelity vs privacy trade-off across evaluated synthesizers."""
    points = []
    for name, res in bench_results.get("detailed", {}).items():
        f_sum = res.get("fidelity", {}).get("summary", {})
        p_dcr = res.get("privacy", {}).get("dcr_and_nndr", {})
        p_mia = res.get("privacy", {}).get("membership_inference", {})

        points.append({
            "model": name,
            "mean_ks": f_sum.get("mean_ks", 1.0),
            "correlation_frobenius": f_sum.get("pearson_frobenius_distance", 1.0),
            "mean_dcr_train": p_dcr.get("mean_dcr_train", 0.0),
            "dcr_ratio": p_dcr.get("dcr_ratio", 0.0),
            "mia_roc_auc": p_mia.get("mia_roc_auc", 0.5),
        })

    return {
        "experiment": "E7_Fidelity_Privacy_Tradeoff",
        "tradeoff_points": points,
    }
