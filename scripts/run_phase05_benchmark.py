"""CLI script to run comprehensive Phase 05 baseline benchmark comparison."""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
import torch

from tabsynth.data.dataset import create_stratified_split
from tabsynth.data.schema import infer_table_schema
from tabsynth.evaluation.benchmarks.copula_wrapper import GaussianCopulaSynthesizer
from tabsynth.evaluation.benchmarks.tvae import TVAESynthesizer
from tabsynth.evaluation.benchmarks.diffusion_wrappers import (
    TabSynthHybridSynthesizer,
    TabDDPMMLPSynthesizer,
)
from tabsynth.evaluation.benchmarks.benchmark_runner import BenchmarkRunner
from tabsynth.evaluation.visualization import (
    plot_marginal_distributions,
    plot_correlation_difference,
    plot_dcr_distributions,
    save_figure,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 05 benchmark comparison.")
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
    print("TabSynth AI - Phase 05 Benchmark Comparison Suite")
    print(f"[*] Dataset:       {args.data}")
    print(f"[*] Target Column: {args.target}")
    print(f"[*] Samples/Model: {args.num_samples:,}")
    print(f"[*] Output Dir:    {args.output_dir}")
    print("=" * 75)

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found at: {args.data}")

    # 1. Load Data and Create Zero-Leakage Split
    df_raw = pd.read_csv(args.data)
    df_train, df_val, df_test = create_stratified_split(df_raw, target_column=args.target, seed=args.seed)
    schema = infer_table_schema(df_train, target_column=args.target)

    print(f"[+] Splits: Train={len(df_train):,}, Val={len(df_val):,}, Test={len(df_test):,}")
    print(f"[+] Features: {len(schema.continuous_columns)} Continuous, {len(schema.categorical_columns)} Categorical.")

    # 2. Instantiate Baseline Models
    synthesizers = {}

    # Baseline 1: Gaussian Copula
    print("\n[*] Initializing Gaussian Copula Baseline...")
    copula_synth = GaussianCopulaSynthesizer(schema=schema, seed=args.seed)
    synthesizers["Gaussian_Copula"] = copula_synth

    # Baseline 2: Native PyTorch TVAE
    print("\n[*] Initializing & Training TVAE Baseline...")
    tvae_synth = TVAESynthesizer(
        schema=schema,
        epochs=15,
        batch_size=256,
        hidden_dim=128,
        latent_dim=32,
        seed=args.seed,
    )
    synthesizers["TVAE"] = tvae_synth

    # Baseline 3: TabDDPM MLP
    mlp_ckpt = "checkpoints/adult_mlp_best.pt"
    if os.path.exists(mlp_ckpt):
        print(f"\n[*] Loading TabDDPM MLP Denoiser from: {mlp_ckpt}")
        synthesizers["TabDDPM_MLP"] = TabDDPMMLPSynthesizer(checkpoint_path=mlp_ckpt, seed=args.seed)
    else:
        print("\n[!] TabDDPM MLP checkpoint not found; training freshly for 5 epochs...")
        mlp_s = TabDDPMMLPSynthesizer(seed=args.seed)
        mlp_s.fit(df_train, target_column=args.target)
        synthesizers["TabDDPM_MLP"] = mlp_s

    # Baseline 4: TabSynth Hybrid (Ours)
    hybrid_ckpt = "checkpoints/adult_hybrid_best.pt"
    if os.path.exists(hybrid_ckpt):
        print(f"\n[*] Loading TabSynth Hybrid Denoiser from: {hybrid_ckpt}")
        synthesizers["TabSynth_Hybrid"] = TabSynthHybridSynthesizer(checkpoint_path=hybrid_ckpt, seed=args.seed)
    else:
        print("\n[!] TabSynth Hybrid checkpoint not found; training freshly for 5 epochs...")
        hyb_s = TabSynthHybridSynthesizer(seed=args.seed)
        hyb_s.fit(df_train, target_column=args.target)
        synthesizers["TabSynth_Hybrid"] = hyb_s

    # 3. Execute Benchmarking Suite
    print("\n" + "=" * 75)
    print("[*] Commencing Multi-Model Benchmark Evaluation...")
    print("=" * 75)

    runner = BenchmarkRunner(
        schema=schema,
        num_samples=args.num_samples,
        run_fidelity=True,
        run_utility=True,
        run_privacy=True,
        subsample_size=1000,
        random_state=args.seed,
    )

    t_bench_start = time.time()
    benchmark_output = runner.run(synthesizers=synthesizers, df_train=df_train, df_test=df_test)
    total_elapsed = time.time() - t_bench_start

    # 4. Display Results
    df_comparison = benchmark_output["comparison_table"]
    markdown_table = benchmark_output["comparison_markdown"]

    print("\n" + "=" * 75)
    print("BENCHMARK COMPARISON RESULTS")
    print("=" * 75)
    print(markdown_table)
    print(f"\n[+] Total benchmark completed in {total_elapsed:.2f}s.")

    # 5. Generate Visualizations for TabSynth Hybrid vs Real
    print("\n[*] Generating Interactive Plotly Visualizations...")
    hybrid_synth_instance = synthesizers.get("TabSynth_Hybrid")
    if hybrid_synth_instance is not None:
        df_synth_hybrid = hybrid_synth_instance.sample(args.num_samples)

        # 1D Marginals
        fig_marginals = plot_marginal_distributions(
            real_df=df_test,
            synth_df=df_synth_hybrid,
            continuous_cols=schema.continuous_columns,
            categorical_cols=schema.categorical_columns,
        )
        save_figure(fig_marginals, args.output_dir, "marginal_overlay_hybrid")

        # Correlation Difference Heatmap
        fig_corr = plot_correlation_difference(
            real_df=df_test,
            synth_df=df_synth_hybrid,
            continuous_cols=schema.continuous_columns,
        )
        save_figure(fig_corr, args.output_dir, "correlation_difference_hybrid")

        # DCR Distributions
        p_res = benchmark_output["detailed"]["TabSynth_Hybrid"].get("privacy", {})
        dcr_info = p_res.get("dcr_and_nndr", {})
        if "raw_distances" in dcr_info:
            d_tr = np.min(dcr_info["raw_distances"]["dist_synth_train"], axis=1)
            d_te = np.min(dcr_info["raw_distances"]["dist_synth_test"], axis=1)
            fig_dcr = plot_dcr_distributions(d_tr, d_te)
            save_figure(fig_dcr, args.output_dir, "dcr_distributions_hybrid")

    # 6. Save Artifacts
    table_csv_path = os.path.join(args.output_dir, "benchmark_comparison.csv")
    df_comparison.to_csv(table_csv_path, index=False)

    summary_md_path = os.path.join(args.output_dir, "benchmark_summary.md")
    with open(summary_md_path, "w") as f:
        f.write("# TabSynth AI - Phase 05 Benchmark Comparison\n\n")
        f.write(markdown_table)
        f.write(f"\n\nTotal benchmark run duration: {total_elapsed:.2f} seconds.\n")

    json_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        # Build clean JSON serializable representation
        cleaned_detailed = {}
        for name, m_res in benchmark_output["detailed"].items():
            cleaned_detailed[name] = {
                "name": name,
                "fit_time_sec": m_res["fit_time_sec"],
                "sample_time_sec": m_res["sample_time_sec"],
                "throughput_rows_per_sec": m_res["throughput_rows_per_sec"],
                "parameter_count": m_res["parameter_count"],
                "fidelity": m_res.get("fidelity", {}).get("summary", {}),
                "utility": {
                    "mean_f1_parity": m_res.get("utility", {}).get("mean_f1_parity", 0.0),
                    "tstr_models": {
                        clf: {
                            "macro_f1": v.get("tstr", {}).get("macro_f1", 0.0),
                            "relative_f1_parity": v.get("relative_f1_parity", 0.0),
                            "minority_recall": v.get("tstr", {}).get("minority_recall", 0.0),
                            "auroc": v.get("tstr", {}).get("auroc", 0.5),
                        }
                        for clf, v in m_res.get("utility", {}).get("tstr", {}).items()
                    },
                },
                "privacy": {
                    "overall_risk": m_res.get("privacy", {}).get("overall_privacy_risk", "UNKNOWN"),
                    "dcr_and_nndr": {
                        k: v for k, v in m_res.get("privacy", {}).get("dcr_and_nndr", {}).items()
                        if k != "raw_distances"
                    },
                    "mia_roc_auc": m_res.get("privacy", {}).get("membership_inference", {}).get("mia_roc_auc", 0.5),
                },
            }

        json.dump({
            "comparison_records": df_comparison.to_dict(orient="records"),
            "detailed": cleaned_detailed,
            "total_elapsed_sec": total_elapsed,
        }, f, indent=2)

    print(f"[+] Benchmark artifacts successfully written to: {args.output_dir}")


if __name__ == "__main__":
    main()
