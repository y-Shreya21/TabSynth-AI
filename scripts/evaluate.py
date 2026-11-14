"""Standalone CLI evaluation and auditing script for synthetic tabular datasets."""

import os
import json
import argparse
import pandas as pd
from tabsynth.data.schema import infer_table_schema
from tabsynth.evaluation.fidelity import evaluate_statistical_fidelity
from tabsynth.evaluation.utility import evaluate_downstream_utility
from tabsynth.evaluation.privacy import evaluate_empirical_privacy


def run_evaluation(args):
    print(f"[*] Loading real test data: {args.real}")
    df_real = pd.read_csv(args.real)
    print(f"[*] Loading synthetic data: {args.synthetic}")
    df_synth = pd.read_csv(args.synthetic)

    schema = infer_table_schema(df_real, target_column=args.target)
    continuous_cols = schema.continuous_columns
    categorical_cols = schema.categorical_columns

    print("[*] 1. Evaluating Statistical Fidelity...")
    fidelity_report = evaluate_statistical_fidelity(
        real_df=df_real,
        synth_df=df_synth,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
    )
    print(f"    - Mean Kolmogorov-Smirnov: {fidelity_report['mean_kolmogorov_smirnov']:.4f}")
    print(f"    - Mean Total Variation Distance: {fidelity_report['mean_total_variation_distance']:.4f}")
    print(f"    - Correlation Frobenius Distance: {fidelity_report['correlation_frobenius_distance']:.4f}")
    print(f"    - C2ST Indistinguishability Accuracy: {fidelity_report['c2st_accuracy']:.4f}")

    utility_report = {}
    privacy_report = {}

    if args.train_data and os.path.exists(args.train_data):
        print(f"[*] Loading real train data: {args.train_data}")
        df_train = pd.read_csv(args.train_data)

        if args.target and args.target in df_real.columns:
            print("[*] 2. Evaluating Downstream Machine Learning Utility (TSTR vs TRTR)...")
            utility_report = evaluate_downstream_utility(
                df_train_real=df_train,
                df_train_synth=df_synth,
                df_test_real=df_real,
                target_column=args.target,
            )
            for clf_name, res in utility_report.items():
                print(f"    - {clf_name} F1 Parity: {res['relative_utility_parity']:.4f} | Minority Recall Delta: {res['minority_recall_delta']:+.4f}")

        print("[*] 3. Evaluating Empirical Privacy & Memorization Risk...")
        privacy_report = evaluate_empirical_privacy(
            df_train=df_train,
            df_test=df_real,
            df_synth=df_synth,
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
            subsample_size=1000,
        )
        print(f"    - Mean DCR Ratio (Train/Test): {privacy_report['dcr_ratio']:.4f}")
        print(f"    - Exact Match Count: {privacy_report['exact_record_matches']}")
        print(f"    - Membership Inference Attack AUROC: {privacy_report['membership_inference_auroc']:.4f} ({privacy_report['privacy_risk_assessment']})")

    final_report = {
        "dataset_name": schema.dataset_name,
        "real_test_samples": len(df_real),
        "synthetic_samples": len(df_synth),
        "fidelity": fidelity_report,
        "utility": utility_report,
        "privacy": privacy_report,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"[+] Full evaluation report serialized to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabSynth AI Evaluation CLI")
    parser.add_argument("--real", type=str, required=True, help="Path to real test CSV")
    parser.add_argument("--synthetic", type=str, required=True, help="Path to synthetic CSV")
    parser.add_argument("--train-data", type=str, default=None, help="Path to real train CSV")
    parser.add_argument("--target", type=str, default=None, help="Target column name")
    parser.add_argument("--output", type=str, default="experiments/results/evaluation_report.json")
    args = parser.parse_args()

    run_evaluation(args)
