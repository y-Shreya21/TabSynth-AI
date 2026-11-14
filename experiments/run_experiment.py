"""Automated research experiment runner for TabSynth AI (Experiments E1 to E9)."""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from tabsynth.data.schema import infer_table_schema
from tabsynth.data.preprocessor import TabularPreprocessor
from tabsynth.data.dataset import create_stratified_split, build_dataloaders
from tabsynth.diffusion.gaussian import GaussianDiffusion
from tabsynth.diffusion.multinomial import MultinomialDiffusion
from tabsynth.models.hybrid import TabSynthHybridModel
from tabsynth.baselines.mlp_diffusion import TabDDPMMLPModel
from tabsynth.training.losses import TabSynthLoss
from tabsynth.training.trainer import TabSynthTrainer, select_device
from tabsynth.sampling.sampler import TabSynthSampler
from tabsynth.evaluation.fidelity import evaluate_statistical_fidelity
from tabsynth.evaluation.utility import evaluate_downstream_utility
from tabsynth.evaluation.privacy import evaluate_empirical_privacy


def run_experiment_e1(data_path: str = "data/raw/adult.csv", target_col: str = "income", epochs: int = 5, seeds: list = [42, 1337]):
    """E1: Architectural Comparison - Hybrid Transformer-MLP vs. MLP-Only Baseline (RQ1 / H1)."""
    print("=" * 70)
    print("EXPERIMENT E1: Hybrid Transformer-MLP vs. MLP-Only Baseline")
    print("=" * 70)

    device = select_device("auto")
    df = pd.read_csv(data_path)
    results = {"hybrid": [], "mlp_only": []}

    for seed in seeds:
        print(f"\n--- Running Seed: {seed} ---")
        df_train, df_val, df_test = create_stratified_split(df, target_column=target_col, seed=seed)
        schema = infer_table_schema(df_train, target_column=target_col)
        preprocessor = TabularPreprocessor(schema=schema, seed=seed).fit(df_train)

        train_loader, val_loader, _ = build_dataloaders(df_train, df_val, df_test, preprocessor, batch_size=256)

        num_cont = len(schema.continuous_columns)
        cards = schema.cardinalities
        num_classes = len(schema.class_distribution) if schema.class_distribution else 2

        for arch_name, is_hybrid in [("hybrid", True), ("mlp_only", False)]:
            print(f"[*] Training {arch_name} model (Seed {seed})...")
            g_diff = GaussianDiffusion(num_timesteps=50, device=device)
            m_diff = MultinomialDiffusion(cardinalities=cards, num_timesteps=50, device=device)

            if is_hybrid:
                model = TabSynthHybridModel(num_cont, cards, num_classes=num_classes, d_model=32, transformer_layers=2, attention_heads=4, mlp_hidden_dim=128)
            else:
                model = TabDDPMMLPModel(num_cont, cards, num_classes=num_classes, d_model=32, mlp_hidden_dim=128)

            loss_fn = TabSynthLoss(m_diff, cards)
            trainer = TabSynthTrainer(model, g_diff, m_diff, loss_fn, train_loader, val_loader, preprocessor, learning_rate=3e-4, device=str(device))
            trainer.fit(epochs=epochs, verbose_interval=epochs)

            # Sample 500 records
            sampler = TabSynthSampler(model, g_diff, m_diff, preprocessor, device=device)
            t0 = time.time()
            synth_df = sampler.sample(num_samples=500, batch_size=256, condition_strategy="balanced", cfg_scale=1.25)
            sample_time = time.time() - t0

            # Evaluate fidelity and utility
            fid = evaluate_statistical_fidelity(df_test, synth_df, schema.continuous_columns, schema.categorical_columns)
            util = evaluate_downstream_utility(df_train, synth_df, df_test, target_col)

            run_record = {
                "seed": seed,
                "sampling_time_sec": sample_time,
                "mean_ks": fid["mean_kolmogorov_smirnov"],
                "mean_tvd": fid["mean_total_variation_distance"],
                "correlation_frobenius": fid["correlation_frobenius_distance"],
                "c2st_accuracy": fid["c2st_accuracy"],
                "rf_f1_parity": util["random_forest"]["relative_utility_parity"],
                "rf_minority_recall_delta": util["random_forest"]["minority_recall_delta"],
            }
            results[arch_name].append(run_record)
            print(f"    -> {arch_name} Corr Frob: {run_record['correlation_frobenius']:.4f} | KS: {run_record['mean_ks']:.4f} | F1 Parity: {run_record['rf_f1_parity']:.4f}")

    os.makedirs("experiments/results", exist_ok=True)
    out_file = "experiments/results/experiment_e1_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Experiment E1 completed. Results saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabSynth AI Research Experiment CLI")
    parser.add_argument("--exp", choices=["e1", "all"], default="e1")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if args.exp in ["e1", "all"]:
        run_experiment_e1(epochs=args.epochs)
