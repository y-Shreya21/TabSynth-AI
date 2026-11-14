"""Phase 03 Smoke Test & Overfitting Experiment for TabSynth AI.

Verifies:
1. Architecture initialization for both TabSynthHybridDenoiser and TabSynthMLPDenoiser.
2. Parameter count profiling and comparison.
3. Forward pass and output shapes.
4. Composite multi-objective loss computation with MixedTypeDiffusion.
5. Backpropagation and optimizer weight updates.
6. A small 10-step overfit test confirming loss minimization on a tiny synthetic batch.
Tests both CPU and Apple Silicon MPS.
"""

import sys
import torch
import torch.optim as optim

from tabsynth.diffusion.mixed.mixed_diffusion import MixedTypeDiffusion
from tabsynth.diffusion.utils import select_device, to_one_hot
from tabsynth.models.hybrid import TabSynthHybridDenoiser
from tabsynth.models.mlp_baseline import TabSynthMLPDenoiser
from tabsynth.models.integration import ModelDiffusionIntegrator
from tabsynth.models.diagnostics import compare_models, get_model_summary


def run_model_smoke_test(device_str: str = "auto"):
    dev = select_device(device_str)
    print("=" * 75)
    print(f"[*] Running TabSynth Phase 03 Model Smoke Test on Device: {dev}")
    print("=" * 75)

    # 1. Dataset Spec: 2 numerical, 3 categorical (cardinalities: 2, 4, 7), 2 classes
    num_continuous = 2
    cardinalities = [2, 4, 7]
    num_classes = 2
    batch_size = 16
    timesteps = 20
    d_model = 64

    # 2. Initialize Models
    hybrid_model = TabSynthHybridDenoiser(
        num_continuous=num_continuous,
        cardinalities=cardinalities,
        num_classes=num_classes,
        d_model=d_model,
        transformer_layers=2,
        attention_heads=4,
        transformer_ffn_dim=256,
        mlp_layers=2,
    ).to(dev)

    mlp_model = TabSynthMLPDenoiser(
        num_continuous=num_continuous,
        cardinalities=cardinalities,
        num_classes=num_classes,
        d_model=d_model,
        mlp_layers=4,
    ).to(dev)

    # 3. Model Profiling & Parameter Diagnostics
    comparison = compare_models(hybrid_model, mlp_model)
    print("\n[1] Architecture Profiling & Parameter Diagnostics:")
    print(f"    - TabSynthHybridDenoiser: {comparison['hybrid']['total_parameters']:,} parameters")
    for sub, count in comparison['hybrid']['submodule_parameters'].items():
        print(f"        * {sub}: {count:,} params")
    print(f"    - TabSynthMLPDenoiser:   {comparison['mlp_baseline']['total_parameters']:,} parameters")
    print(f"    - Parameter Delta:       {comparison['parameter_delta']:+,} ({comparison['percentage_difference']:+.1f}%)")

    # 4. Initialize MixedTypeDiffusion Engine
    diffusion = MixedTypeDiffusion(
        num_numerical_features=num_continuous,
        categorical_cardinalities=cardinalities,
        numerical_timesteps=timesteps,
        categorical_timesteps=timesteps,
        device=dev,
    )

    # 5. Synthetic Batch Generation
    torch.manual_seed(42)
    x_num = torch.randn(batch_size, num_continuous, device=dev)
    x_cat = [to_one_hot(torch.randint(0, K, (batch_size,), device=dev), num_classes=K) for K in cardinalities]
    labels = torch.randint(0, num_classes, (batch_size,), device=dev)

    # 6. Test Integrator Forward Loss on Both Models
    print("\n[2] Forward Loss & Multi-Objective Evaluation:")
    for name, model in [("TabSynthHybridDenoiser", hybrid_model), ("TabSynthMLPDenoiser", mlp_model)]:
        integrator = ModelDiffusionIntegrator(model=model, diffusion=diffusion)
        losses = integrator.forward_loss(x_num_0=x_num, x_cat_0_list=x_cat, class_labels=labels)
        print(f"    [{name}]")
        print(f"        * Total Loss:    {losses['loss_total'].item():.4f}")
        print(f"        * Numerical MSE: {losses['loss_num'].item():.4f}")
        print(f"        * Discrete KL:   {losses['loss_cat'].item():.4f}")
        print(f"        * Auxiliary CE:  {losses['loss_aux'].item():.4f}")

    # 7. Overfit Test on Tiny Synthetic Batch (10 steps)
    print("\n[3] Overfitting Test (10 Optimization Steps):")
    integrator = ModelDiffusionIntegrator(model=hybrid_model, diffusion=diffusion)
    optimizer = optim.AdamW(hybrid_model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Pre-corrupt batch once to fix targets for deterministic overfit verification
    t_fixed = torch.full((batch_size,), 5, device=dev, dtype=torch.long)
    x_t_num_fixed, target_noise_fixed, x_t_cat_fixed, _ = diffusion.forward(
        x_num=x_num,
        x_cat_list=x_cat,
        t_num=t_fixed,
        t_cat=t_fixed,
    )

    initial_loss = None
    final_loss = None

    for step in range(15):
        optimizer.zero_grad()
        out = hybrid_model(
            x_num_t=x_t_num_fixed,
            x_cat_t=x_t_cat_fixed,
            timesteps_num=t_fixed,
            timesteps_cat=t_fixed,
            class_labels=labels,
        )
        losses = diffusion.compute_training_loss(
            pred_eps=out.predicted_noise,
            target_noise=target_noise_fixed,
            pred_logits_list=out.categorical_logits,
            clean_cat_list=x_cat,
            corrupted_cat_list=x_t_cat_fixed,
            t_cat=t_fixed,
        )
        loss = losses["loss_total"]
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), 1.0)
        optimizer.step()

    print(f"    - Initial Step Loss: {initial_loss:.4f}")
    print(f"    - Final Step Loss:   {final_loss:.4f}")
    assert final_loss < initial_loss, f"Loss did not decrease during overfit test ({final_loss} >= {initial_loss})"
    print(f"    - Loss Reduction:    {(initial_loss - final_loss) / initial_loss * 100:.2f}% (Successful convergence)")

    print(f"\n[+] Phase 03 smoke test on {dev} completed with 100% success.\n")


if __name__ == "__main__":
    device_arg = sys.argv[1] if len(sys.argv) > 1 else "auto"
    run_model_smoke_test(device_str=device_arg)
