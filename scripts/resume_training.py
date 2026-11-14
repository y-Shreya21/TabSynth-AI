"""Script to resume training a TabSynth AI model from a saved checkpoint."""

import os
import argparse
import pandas as pd
import torch

from tabsynth.data.dataset import create_stratified_split, build_dataloaders
from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.models.hybrid import TabSynthHybridDenoiser
from tabsynth.models.mlp_baseline import TabSynthMLPDenoiser
from tabsynth.training.checkpoint import CheckpointManager
from tabsynth.training.trainer import TabSynthTrainer, select_device
from tabsynth.training.losses import TabSynthLoss
from tabsynth.training.optimizer import build_optimizer
from tabsynth.training.scheduler import build_scheduler
from tabsynth.training.early_stopping import EarlyStopping


def main():
    parser = argparse.ArgumentParser(description="Resume training from an existing checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (.pt).")
    parser.add_argument("--data", type=str, default="", help="Path to dataset (if different from config).")
    parser.add_argument("--additional_epochs", type=int, default=10, help="Number of additional epochs to train.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"[*] Resuming from checkpoint: {args.checkpoint} on {device}")

    ckpt = CheckpointManager.load(args.checkpoint, device=device)
    preprocessor = ckpt["preprocessor"]
    start_epoch = ckpt.get("epoch", 0)
    cfg = ckpt.get("config", {}) or {}

    schema = preprocessor.schema
    data_path = args.data or cfg.get("data", {}).get("raw_data_path", "data/raw/adult.csv")
    df = pd.read_csv(data_path)

    df_train, df_val, df_test = create_stratified_split(df, target_column=schema.target_column, seed=42)
    batch_size = cfg.get("data", {}).get("batch_size", 256)

    train_loader, val_loader, _ = build_dataloaders(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        preprocessor=preprocessor,
        batch_size=batch_size,
    )

    # Recreate model
    num_continuous = len(schema.continuous_columns)
    cardinalities = schema.cardinalities
    num_classes = len(schema.class_distribution) if schema.class_distribution else 2
    model_type = cfg.get("model", {}).get("type", "hybrid_transformer_mlp")
    d_model = cfg.get("model", {}).get("d_model", 64)

    if model_type in ("hybrid_transformer_mlp", "hybrid"):
        model = TabSynthHybridDenoiser(
            num_continuous=num_continuous,
            cardinalities=cardinalities,
            num_classes=num_classes,
            d_model=d_model,
        )
    else:
        model = TabSynthMLPDenoiser(
            num_continuous=num_continuous,
            cardinalities=cardinalities,
            num_classes=num_classes,
            d_model=d_model,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Setup diffusion
    num_timesteps = cfg.get("diffusion", {}).get("steps", 100)
    gaussian_diff = GaussianDiffusion(num_timesteps=num_timesteps, device=device)
    multinomial_diff = CategoricalDiffusion(cardinalities=cardinalities, num_timesteps=num_timesteps, device=device)
    loss_fn = TabSynthLoss(multinomial_diffusion=multinomial_diff, cardinalities=cardinalities)

    # Recreate optimizer and restore state
    optimizer = build_optimizer(model=model, learning_rate=cfg.get("training", {}).get("learning_rate", 2e-4))
    if ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    trainer = TabSynthTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        preprocessor=preprocessor,
        gaussian_diffusion=gaussian_diff,
        multinomial_diffusion=multinomial_diff,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=cfg.get("training", {}).get("checkpoint", {}).get("save_dir", "checkpoints/"),
        config=cfg,
    )

    print(f"[+] Resuming training from epoch {start_epoch + 1} to {start_epoch + args.additional_epochs}...")
    trainer.fit(epochs=args.additional_epochs, prefix=cfg.get("training", {}).get("checkpoint", {}).get("prefix", "resumed"))
    print("[+] Resumed training session finished.")


if __name__ == "__main__":
    main()
