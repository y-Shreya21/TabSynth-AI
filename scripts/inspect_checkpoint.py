"""Utility script to inspect saved TabSynth AI checkpoint files."""

import os
import argparse
import torch

from tabsynth.training.checkpoint import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description="Inspect TabSynth AI model checkpoint bundle.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (.pt).")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    file_size_mb = os.path.getsize(args.checkpoint) / (1024 * 1024)
    print("=" * 70)
    print("TabSynth AI - Checkpoint Inspector")
    print(f"[*] Path:      {args.checkpoint}")
    print(f"[*] File Size: {file_size_mb:.2f} MB")
    print("=" * 70)

    ckpt = CheckpointManager.load(args.checkpoint, device="cpu")

    print(f"[+] Saved Epoch:       {ckpt.get('epoch', 'Unknown')}")
    print(f"[+] Validation Loss:   {ckpt.get('val_loss', 'Unknown')}")

    model_state = ckpt.get("model_state_dict", {})
    total_params = sum(p.numel() for p in model_state.values())
    print(f"[+] Model Parameters:  {total_params:,} across {len(model_state)} weight tensors")

    opt_state = ckpt.get("optimizer_state_dict")
    if opt_state is not None:
        param_groups = opt_state.get("param_groups", [])
        lrs = [g.get("lr", "N/A") for g in param_groups]
        print(f"[+] Optimizer State:   Present ({len(param_groups)} parameter groups, lr={lrs})")
    else:
        print("[+] Optimizer State:   Not present")

    sched_state = ckpt.get("scheduler_state_dict")
    print(f"[+] Scheduler State:   {'Present' if sched_state is not None else 'Not present'}")

    preprocessor = ckpt.get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "schema"):
        schema = preprocessor.schema
        print("\n--- Preprocessor & Dataset Schema ---")
        print(f"Target Column:        {schema.target_column}")
        print(f"Continuous Columns:   ({len(schema.continuous_columns)}) {schema.continuous_columns}")
        print(f"Categorical Columns:  ({len(schema.categorical_columns)}) {schema.categorical_columns}")
        print(f"Cardinalities:        {schema.cardinalities}")
        if schema.class_distribution:
            print(f"Class Distribution:   {schema.class_distribution}")
        if hasattr(preprocessor, "target_to_idx"):
            print(f"Target Mapping:       {preprocessor.target_to_idx}")
    else:
        print("\n[!] Preprocessor not found in checkpoint.")

    cfg = ckpt.get("config")
    if cfg:
        print("\n--- Configuration Summary ---")
        model_cfg = cfg.get("model", {})
        diff_cfg = cfg.get("diffusion", {})
        print(f"Model Type:           {model_cfg.get('type', 'N/A')}")
        print(f"Embedding Dim (d_model): {model_cfg.get('d_model', 'N/A')}")
        print(f"Diffusion Steps:      {diff_cfg.get('steps', 'N/A')}")


if __name__ == "__main__":
    main()
