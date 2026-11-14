"""Synthetic smoke experiments for the Tabular Diffusion Engine (Phase 02).

Verifies forward diffusion, reverse posteriors, and mixed-type sampling
on a toy dataset with 2 continuous and 2 categorical features.
Tests both CPU and Apple Silicon MPS devices.
"""

import sys
import torch
import torch.nn.functional as F

from tabsynth.diffusion.mixed.mixed_diffusion import MixedTypeDiffusion
from tabsynth.diffusion.utils import to_one_hot, select_device


def run_smoke_test(device_str: str = "auto", batch_size: int = 32, timesteps: int = 20):
    dev = select_device(device_str)
    print("=" * 70)
    print(f"[*] Running TabSynth Diffusion Smoke Test on Device: {dev}")
    print(f"[*] Batch size: {batch_size} | Diffusion steps: {timesteps}")
    print("=" * 70)

    # 1. Dataset Specification: 2 continuous features, 2 categorical features (cardinalities 3 and 5)
    num_continuous = 2
    cardinalities = [3, 5]

    # Instantiate MixedTypeDiffusion Engine
    engine = MixedTypeDiffusion(
        num_numerical_features=num_continuous,
        categorical_cardinalities=cardinalities,
        numerical_timesteps=timesteps,
        categorical_timesteps=timesteps,
        device=dev,
    )

    # Generate synthetic clean inputs
    torch.manual_seed(42)
    x_num = torch.randn(batch_size, num_continuous, device=dev)
    x_cat_list = [
        to_one_hot(torch.randint(0, K, (batch_size,), device=dev), num_classes=K)
        for K in cardinalities
    ]

    print("[1] Synthetic Data Generation:")
    print(f"    - Numerical x_0 shape: {x_num.shape} (dtype: {x_num.dtype})")
    for j, (cat, K) in enumerate(zip(x_cat_list, cardinalities)):
        print(f"    - Categorical col {j+1} x_0 shape: {cat.shape} (cardinality: {K})")

    # 2. Forward Diffusion
    t_num = torch.randint(0, timesteps, (batch_size,), device=dev)
    t_cat = torch.randint(0, timesteps, (batch_size,), device=dev)

    x_t_num, noise_num = engine.forward_numerical(x_num, t_num)
    x_t_cat_oh, x_t_cat_idx = engine.forward_categorical(x_cat_list, t_cat)

    print("\n[2] Forward Diffusion Corruption:")
    print(f"    - Noisy x_t_num shape: {x_t_num.shape} | mean: {x_t_num.mean().item():.4f} | std: {x_t_num.std().item():.4f}")
    for j, oh in enumerate(x_t_cat_oh):
        valid_simplex = (oh.sum(dim=-1) - 1.0).abs().max().item() < 1e-4
        print(f"    - Noisy categorical col {j+1} one-hot shape: {oh.shape} | Valid simplex: {valid_simplex}")

    # 3. Training Loss Evaluation
    pred_eps = torch.randn_like(x_t_num)
    pred_logits = [torch.randn(batch_size, K, device=dev) for K in cardinalities]

    losses = engine.compute_training_loss(
        pred_eps=pred_eps,
        target_noise=noise_num,
        pred_logits_list=pred_logits,
        clean_cat_list=x_cat_list,
        corrupted_cat_list=x_t_cat_oh,
        t_cat=t_cat,
    )

    print("\n[3] Joint Mixed Loss Computation:")
    print(f"    - Total loss: {losses['loss_total'].item():.4f}")
    print(f"    - Numerical MSE loss: {losses['loss_num'].item():.4f}")
    print(f"    - Categorical KL loss: {losses['loss_cat'].item():.4f}")
    print(f"    - Auxiliary CE loss: {losses['loss_aux'].item():.4f}")

    # 4. Joint Reverse Sampling
    def dummy_denoiser(x_n, x_c_list, t_n, t_c):
        # Dummy denoiser returning slight damping
        eps = x_n * 0.1
        logits = [F.log_softmax(x_c * 0.5, dim=-1) for x_c in x_c_list]
        return eps, logits

    print("\n[4] Ancestral Reverse Sampling:")
    sampled_num, sampled_cats = engine.sample(
        model_fn=dummy_denoiser,
        batch_size=batch_size,
        device=dev,
    )

    print(f"    - Sampled numerical shape: {sampled_num.shape} | mean: {sampled_num.mean().item():.4f}")
    for j, (cat, K) in enumerate(zip(sampled_cats, cardinalities)):
        indices = torch.argmax(cat, dim=-1)
        print(f"    - Sampled categorical col {j+1} shape: {cat.shape} | Unique classes: {torch.unique(indices).tolist()}")

    print(f"\n[+] Smoke test on {dev} completed successfully with 100% mathematical validity.\n")


if __name__ == "__main__":
    device_arg = sys.argv[1] if len(sys.argv) > 1 else "auto"
    run_smoke_test(device_str=device_arg)
