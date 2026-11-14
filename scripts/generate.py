"""CLI script for conditional synthetic tabular data generation with Classifier-Free Guidance."""

import os
import argparse
import time
import pandas as pd
import torch

from tabsynth.diffusion.utils import select_device
from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.models.hybrid import TabSynthHybridDenoiser
from tabsynth.models.mlp_baseline import TabSynthMLPDenoiser
from tabsynth.training.checkpoint import CheckpointManager
from tabsynth.training.seed import seed_everything
from tabsynth.sampling.generation import GenerationPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic tabular data from trained checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt).")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of synthetic rows to generate.")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch generation chunk size.")
    parser.add_argument("--output", type=str, default="data/synthetic/synthetic_output.csv", help="Destination CSV path.")
    parser.add_argument(
        "--class_condition",
        type=str,
        default="balanced",
        choices=["balanced", "natural", "class_specific", "unconditional"],
        help="Class-conditioning strategy.",
    )
    parser.add_argument("--specific_class", type=int, default=0, help="Target class when using class_specific strategy.")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-Free Guidance scale w.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    print("=" * 70)
    print("TabSynth AI - Conditional Synthetic Data Generation Engine")
    print(f"[*] Checkpoint:           {args.checkpoint}")
    print(f"[*] Samples to generate:  {args.num_samples:,}")
    print(f"[*] Condition Strategy:   {args.class_condition}")
    print(f"[*] CFG Guidance Scale:   w = {args.cfg_scale:.2f}")
    print(f"[*] Destination File:     {args.output}")
    print("=" * 70)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    device = select_device(args.device)
    print(f"[+] Using compute hardware: {device} (Apple MPS: {torch.backends.mps.is_available()})")

    # 1. Load Checkpoint Metadata & State
    ckpt = CheckpointManager.load(checkpoint_path=args.checkpoint, device=device)
    preprocessor = ckpt["preprocessor"]
    cfg = ckpt.get("config", {}) or {}

    schema = preprocessor.schema
    num_continuous = len(schema.continuous_columns)
    cardinalities = schema.cardinalities
    num_classes = len(schema.class_distribution) if schema.class_distribution else 2

    # 2. Reconstruct Model Architecture
    model_type = cfg.get("model", {}).get("type", "hybrid_transformer_mlp")
    d_model = cfg.get("model", {}).get("d_model", 64)
    p_uncond = cfg.get("model", {}).get("conditioning", {}).get("p_uncond", 0.15)

    if model_type in ("hybrid_transformer_mlp", "hybrid"):
        t_layers = cfg.get("model", {}).get("transformer", {}).get("layers", 2)
        heads = cfg.get("model", {}).get("transformer", {}).get("heads", 4)
        t_ffn = cfg.get("model", {}).get("transformer", {}).get("dim_feedforward", 256)
        mlp_layers = cfg.get("model", {}).get("denoiser_mlp", {}).get("layers", 2)
        mlp_dim = cfg.get("model", {}).get("denoiser_mlp", {}).get("hidden_dim", 256)

        model = TabSynthHybridDenoiser(
            num_continuous=num_continuous,
            cardinalities=cardinalities,
            num_classes=num_classes,
            d_model=d_model,
            transformer_layers=t_layers,
            attention_heads=heads,
            transformer_ffn_dim=t_ffn,
            mlp_hidden_dim=mlp_dim,
            mlp_layers=mlp_layers,
            p_uncond=p_uncond,
        )
    else:
        mlp_layers = cfg.get("model", {}).get("denoiser_mlp", {}).get("layers", 3)
        mlp_dim = cfg.get("model", {}).get("denoiser_mlp", {}).get("hidden_dim", 256)

        model = TabSynthMLPDenoiser(
            num_continuous=num_continuous,
            cardinalities=cardinalities,
            num_classes=num_classes,
            d_model=d_model,
            mlp_hidden_dim=mlp_dim,
            mlp_layers=mlp_layers,
            p_uncond=p_uncond,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[+] Restored model weights from epoch {ckpt.get('epoch', '?')} (val_loss: {ckpt.get('val_loss', 0.0):.4f})")

    # 3. Setup Diffusion Engines
    num_timesteps = cfg.get("diffusion", {}).get("steps", 100)
    sched_num = cfg.get("diffusion", {}).get("numerical_schedule", "cosine")
    sched_cat = cfg.get("diffusion", {}).get("categorical_schedule", "cosine")

    gaussian_diff = GaussianDiffusion(
        num_timesteps=num_timesteps,
        schedule_type=sched_num,
        device=device,
    )
    multinomial_diff = CategoricalDiffusion(
        cardinalities=cardinalities,
        num_timesteps=num_timesteps,
        schedule_type=sched_cat,
        device=device,
    )

    # 4. Instantiate Generation Pipeline
    pipeline = GenerationPipeline(
        model=model,
        preprocessor=preprocessor,
        gaussian_diffusion=gaussian_diff,
        multinomial_diffusion=multinomial_diff,
        num_timesteps=num_timesteps,
        device=device,
    )

    # 5. Synthesize Tabular Data
    t0 = time.time()
    synthetic_df = pipeline.generate(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        condition_strategy=args.class_condition,
        specific_class=args.specific_class,
        cfg_scale=args.cfg_scale,
        clip_denoised=True,
    )
    elapsed = time.time() - t0

    print(f"[+] Synthesis finished in {elapsed:.2f}s ({len(synthetic_df) / max(1e-6, elapsed):.1f} rows/sec).")

    # 6. Save Output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    synthetic_df.to_csv(args.output, index=False)
    print(f"[+] Successfully saved {len(synthetic_df)} records to: {args.output}")

    # Summary
    print("\n--- Synthetic Sample Preview ---")
    print(synthetic_df.head(3))
    if schema.target_column in synthetic_df.columns:
        print("\n--- Target Column Distribution ---")
        print(synthetic_df[schema.target_column].value_counts(normalize=True))


if __name__ == "__main__":
    main()
