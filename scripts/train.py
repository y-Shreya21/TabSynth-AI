"""Standalone CLI training script for TabSynth AI and baselines."""

import os
import argparse
import yaml
import pandas as pd
import torch

from tabsynth.data.schema import infer_table_schema
from tabsynth.data.preprocessor import TabularPreprocessor
from tabsynth.data.dataset import create_stratified_split, build_dataloaders
from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.diffusion.mixed.mixed_diffusion import MixedTypeDiffusion
from tabsynth.models.hybrid import TabSynthHybridDenoiser, TabSynthHybridModel
from tabsynth.models.mlp_baseline import TabSynthMLPDenoiser, TabDDPMMLPModel
from tabsynth.models.diagnostics import count_parameters
from tabsynth.training.losses import TabSynthLoss
from tabsynth.training.optimizer import build_optimizer
from tabsynth.training.scheduler import build_scheduler
from tabsynth.training.early_stopping import EarlyStopping
from tabsynth.training.seed import seed_everything
from tabsynth.training.trainer import TabSynthTrainer, select_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train TabSynth AI or Baseline Diffusion Models.")
    parser.add_argument("--config", type=str, default="", help="Path to YAML experiment configuration.")
    parser.add_argument("--data", type=str, default="", help="Path to raw CSV dataset.")
    parser.add_argument("--target", type=str, default="", help="Target column name for conditional generation.")
    parser.add_argument("--epochs", type=int, default=0, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=0, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=0.0, help="Learning rate.")
    parser.add_argument("--model_type", type=str, default="", choices=["", "hybrid_transformer_mlp", "mlp_baseline"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience.")
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--conditional", action="store_true", default=True, help="Train in class-conditional mode.")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Resolve parameters prioritizing CLI flags over YAML
    seed = args.seed or cfg.get("experiment", {}).get("seed", 42)
    seed_everything(seed)

    data_path = args.data or cfg.get("data", {}).get("raw_data_path", "data/raw/adult.csv")
    target_col = args.target or cfg.get("data", {}).get("target_column", "income")
    epochs = args.epochs or cfg.get("training", {}).get("total_epochs", 20)
    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 256)
    lr = args.lr or cfg.get("training", {}).get("learning_rate", 2e-4)
    model_type = args.model_type or cfg.get("model", {}).get("type", "hybrid_transformer_mlp")
    device_name = args.device or cfg.get("training", {}).get("device", "auto")
    ckpt_dir = args.save_dir or cfg.get("training", {}).get("checkpoint", {}).get("save_dir", "checkpoints/")
    prefix = cfg.get("training", {}).get("checkpoint", {}).get("prefix", "tabsynth")
    patience = args.patience or cfg.get("training", {}).get("early_stopping", {}).get("patience", 10)

    print("=" * 70)
    print("TabSynth AI - Diffusion Model Training Pipeline")
    print(f"[*] Configuration: {args.config or 'CLI Arguments'}")
    print(f"[*] Dataset:       {data_path}")
    print(f"[*] Target Column: {target_col}")
    print(f"[*] Model Type:    {model_type}")
    print(f"[*] Epochs:        {epochs} | Batch Size: {batch_size} | LR: {lr:.2e}")
    print("=" * 70)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Run scripts/download_datasets.py first.")

    df = pd.read_csv(data_path)
    print(f"[+] Loaded raw dataset: {len(df):,} records, {len(df.columns)} columns.")

    # 1. Zero-Leakage Stratified Partitioning & Preprocessor
    df_train, df_val, df_test = create_stratified_split(df, target_column=target_col, seed=seed)
    schema = infer_table_schema(df_train, target_column=target_col)

    preprocessor = TabularPreprocessor(schema=schema, seed=seed).fit(df_train)

    train_loader, val_loader, test_loader = build_dataloaders(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        preprocessor=preprocessor,
        batch_size=batch_size,
    )
    print(f"[+] DataLoaders: Train={len(train_loader)} batches | Val={len(val_loader)} batches | Test={len(test_loader)} batches")

    # 2. Hardware Resolution
    device = select_device(device_name)
    print(f"[+] Target Hardware: {device} (Apple MPS Available: {torch.backends.mps.is_available()})")

    # 3. Diffusion Engines
    diff_cfg = cfg.get("diffusion", {})
    num_timesteps = diff_cfg.get("steps", 100)
    sched_num = diff_cfg.get("numerical_schedule", "cosine")
    sched_cat = diff_cfg.get("categorical_schedule", "cosine")

    gaussian_diff = GaussianDiffusion(
        num_timesteps=num_timesteps,
        schedule_type=sched_num,
        device=device,
    )
    multinomial_diff = CategoricalDiffusion(
        cardinalities=schema.cardinalities,
        num_timesteps=num_timesteps,
        schedule_type=sched_cat,
        device=device,
    )
    mixed_diff = MixedTypeDiffusion(
        num_numerical_features=len(schema.continuous_columns),
        categorical_cardinalities=schema.cardinalities,
        numerical_timesteps=num_timesteps,
        categorical_timesteps=num_timesteps,
        numerical_schedule=sched_num,
        categorical_schedule=sched_cat,
        lambda_num=diff_cfg.get("lambda_num", 1.0),
        lambda_cat=diff_cfg.get("lambda_cat", 0.2),
        lambda_aux=diff_cfg.get("lambda_aux", 0.1),
        device=device,
    )

    # 4. Denoising Neural Architecture
    num_classes = len(schema.class_distribution) if schema.class_distribution else 2
    d_model = cfg.get("model", {}).get("d_model", 64)
    p_uncond = cfg.get("model", {}).get("conditioning", {}).get("p_uncond", 0.15)

    if model_type in ("hybrid_transformer_mlp", "hybrid"):
        t_layers = cfg.get("model", {}).get("transformer", {}).get("layers", 2)
        heads = cfg.get("model", {}).get("transformer", {}).get("heads", 4)
        t_ffn = cfg.get("model", {}).get("transformer", {}).get("dim_feedforward", 256)
        mlp_layers = cfg.get("model", {}).get("denoiser_mlp", {}).get("layers", 2)
        mlp_dim = cfg.get("model", {}).get("denoiser_mlp", {}).get("hidden_dim", 256)

        model = TabSynthHybridDenoiser(
            num_continuous=len(schema.continuous_columns),
            cardinalities=schema.cardinalities,
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
            num_continuous=len(schema.continuous_columns),
            cardinalities=schema.cardinalities,
            num_classes=num_classes,
            d_model=d_model,
            mlp_hidden_dim=mlp_dim,
            mlp_layers=mlp_layers,
            p_uncond=p_uncond,
        )

    param_summary = count_parameters(model)
    print(f"[+] Instantiated {model_type} with {param_summary['total']:,} parameters ({param_summary['trainable']:,} trainable).")

    # 5. Optimization & Training Schedule
    optimizer = build_optimizer(
        model=model,
        optimizer_type=cfg.get("training", {}).get("optimizer", "adamw"),
        learning_rate=lr,
        weight_decay=cfg.get("training", {}).get("weight_decay", 1e-4),
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type=cfg.get("training", {}).get("scheduler", "cosine"),
        total_epochs=epochs,
        warmup_epochs=cfg.get("training", {}).get("warmup_epochs", 2),
        min_lr=cfg.get("training", {}).get("min_lr", 1e-6),
    )

    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=cfg.get("training", {}).get("early_stopping", {}).get("min_delta", 1e-4),
        restore_best_weights=True,
    )

    loss_fn = TabSynthLoss(
        multinomial_diffusion=multinomial_diff,
        cardinalities=schema.cardinalities,
        lambda_num=diff_cfg.get("lambda_num", 1.0),
        lambda_cat=diff_cfg.get("lambda_cat", 0.2),
        lambda_aux=diff_cfg.get("lambda_aux", 0.1),
    )

    trainer = TabSynthTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        preprocessor=preprocessor,
        gaussian_diffusion=gaussian_diff,
        multinomial_diffusion=multinomial_diff,
        mixed_diffusion=mixed_diff,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        max_grad_norm=cfg.get("training", {}).get("max_grad_norm", 1.0),
        device=device,
        checkpoint_dir=ckpt_dir,
        config=cfg,
    )

    # 6. Fit Model
    print(f"[*] Commencing training for {epochs} epochs...")
    history = trainer.fit(epochs=epochs, verbose_interval=1, prefix=prefix)

    best_ckpt = trainer.checkpoint_manager.get_best_checkpoint(prefix=prefix)
    print(f"[+] Training completed successfully. Best checkpoint saved at: {best_ckpt}")
    return best_ckpt


if __name__ == "__main__":
    main()
