"""CLI script to run research ablation experiments (E1, E5, E6, E7) for Phase 05."""

import os
import json
import time
import argparse
import pandas as pd
import numpy as np

from tabsynth.data.dataset import create_stratified_split
from tabsynth.data.schema import infer_table_schema
from tabsynth.evaluation.benchmarks.diffusion_wrappers import (
    TabSynthHybridSynthesizer,
    TabDDPMMLPSynthesizer,
)
from tabsynth.evaluation.experiments.ablations import (
    run_e1_architecture_ablation,
    run_e5_class_sampling_ablation,
    run_e6_minority_augmentation_study,
    run_e7_fidelity_privacy_tradeoff,
)
from tabsynth.evaluation.visualization import plot_augmentation_curves, save_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 05 ablation studies.")
    parser.add_argument("--data", type=str, default="data/raw/adult.csv")
    parser.add_argument("--target", type=str, default="income")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="artifacts/phase05_benchmark")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 75)
    print("TabSynth AI - Phase 05 Research Ablation Studies (E1, E5, E6, E7)")
    print(f"[*] Dataset:       {args.data}")
    print(f"[*] Target Column: {args.target}")
    print(f"[*] Output Dir:    {args.output_dir}")
    print("=" * 75)

    # 1. Load Data
    df_raw = pd.read_csv(args.data)
    df_train, df_val, df_test = create_stratified_split(df_raw, target_column=args.target, seed=args.seed)
    schema = infer_table_schema(df_train, target_column=args.target)

    # 2. Load Checkpointed Models
    hybrid_ckpt = "checkpoints/adult_hybrid_best.pt"
    mlp_ckpt = "checkpoints/adult_mlp_best.pt"

    print("[*] Loading trained hybrid and MLP models...")
    hybrid_synth = TabSynthHybridSynthesizer(checkpoint_path=hybrid_ckpt, seed=args.seed)
    mlp_synth = TabDDPMMLPSynthesizer(checkpoint_path=mlp_ckpt, seed=args.seed)

    ablation_summary: dict = {}

    # 3. E1: Architecture Ablation (Hybrid vs MLP)
    print("\n" + "-" * 60)
    print("[*] Running E1: Architecture Ablation (Cross-Feature Attention)...")
    print("-" * 60)
    e1_results = run_e1_architecture_ablation(
        hybrid_synth=hybrid_synth,
        mlp_synth=mlp_synth,
        df_train=df_train,
        df_test=df_test,
        schema=schema,
        num_samples=args.num_samples,
    )
    ablation_summary["E1_architecture"] = e1_results
    print(f"[✓] E1 Complete: Hybrid F1 Parity: {e1_results['hybrid']['mean_f1_parity']:.4f} vs MLP: {e1_results['mlp']['mean_f1_parity']:.4f}")
    print(f"    Pearson Frobenius Error: Hybrid: {e1_results['hybrid']['fidelity']['pearson_frobenius_distance']:.4f} vs MLP: {e1_results['mlp']['fidelity']['pearson_frobenius_distance']:.4f}")

    # 4. E5: Class Sampling Strategy Ablation
    print("\n" + "-" * 60)
    print("[*] Running E5: Class Sampling Strategy Ablation (Balanced vs Natural)...")
    print("-" * 60)
    e5_results = run_e5_class_sampling_ablation(
        synth=hybrid_synth,
        df_train=df_train,
        df_test=df_test,
        schema=schema,
        num_samples=args.num_samples,
    )
    ablation_summary["E5_class_sampling"] = e5_results
    print(f"[✓] E5 Complete: Natural RF Minority Recall: {e5_results['natural_rf_minority_recall']:.4f} vs Balanced: {e5_results['balanced_rf_minority_recall']:.4f}")
    print(f"    Minority Recall Gain: +{e5_results['minority_recall_gain_from_balancing']:.4f}")

    # 5. E6: Minority Augmentation Study
    print("\n" + "-" * 60)
    print("[*] Running E6: Minority-Class Augmentation Study ([0.25x, 0.5x, 1.0x, 2.0x])...")
    print("-" * 60)
    e6_results = run_e6_minority_augmentation_study(
        synth=hybrid_synth,
        df_train=df_train,
        df_test=df_test,
        schema=schema,
        ratios=[0.25, 0.5, 1.0, 2.0],
        num_samples=args.num_samples,
    )
    ablation_summary["E6_augmentation"] = e6_results
    aug_plot = plot_augmentation_curves(e6_results["augmentation_results"])
    save_figure(aug_plot, args.output_dir, "augmentation_tradeoff_curves")
    print(f"[✓] E6 Complete: Visualized augmentation trade-off across ratios.")

    # 6. Save Combined Ablation Results
    out_json = os.path.join(args.output_dir, "ablation_results.json")
    with open(out_json, "w") as f:
        json.dump(ablation_summary, f, indent=2)

    print("\n" + "=" * 75)
    print(f"[SUCCESS] Ablation studies completed. Results written to: {out_json}")
    print("=" * 75)


if __name__ == "__main__":
    main()
