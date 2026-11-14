# TabSynth AI: Phase 04 Compatibility & Architecture Audit Report

**Date:** 2026-11-15  
**Hardware Environment:** Apple Silicon M4 (Darwin 27.0.1, arm64, 16 GB Unified Memory)  
**Software Environment:** Python 3.11.15, PyTorch 2.15.0 with native Apple MPS (`torch.device("mps")`) and CPU fallback  
**Status:** Audit Completed – Ready for Phase 04 Execution  

---

## 1. Executive Summary

This report establishes the baseline architecture, mathematical interface contracts, tensor shapes, conditioning invariants, and backward-compatibility guarantees for **Phase 04: Training Pipeline and Conditional Synthetic Data Generation**.

The audit inspected all components delivered across Phase 01 (Data Acquisition, Zero-Leakage Preprocessing, EDA), Phase 02 (Gaussian & D3PM Multinomial Diffusion Engine), and Phase 03 (Hybrid Transformer-MLP Denoising Architecture and MLP Baseline). All 67 existing automated tests in `tests/` currently pass without error or warning.

---

## 2. Component Interface Verification

### 2.1 Phase 01: Preprocessing & Data Layer
- **Module:** `src/tabsynth/data/preprocessor.py` (`TabularPreprocessor`) and `src/tabsynth/data/dataset.py` (`TabularDataset`, `build_dataloaders`).
- **Data Invariant:** Preprocessor is fitted strictly on $D_{\text{train}}$. Validation ($D_{\text{val}}$) and test ($D_{\text{test}}$) splits are transformed using fitted transformers without updating statistics (zero data leakage).
- **Output Shapes:**
  - `x_num`: $\mathbb{R}^{B \times D_{\text{num}}}$ continuous features (`float32`), standardized via `QuantileTransformer(output_distribution="normal")` and clipped to $[-5.0, 5.0]$.
  - `x_cat_list`: List of $D_{\text{cat}}$ one-hot tensors, each of shape $\mathbb{R}^{B \times K_j}$ (`float32`), where $K_j$ is the cardinality of categorical column $j$ (including reserved `__MISSING__` and `__UNKNOWN__` tokens).
  - `labels`: $\mathbb{Z}^{B}$ target class indices (`int64` / `torch.long`), strictly separated from feature tensors.
- **Inverse Transformation:** `preprocessor.inverse_transform(x_num, x_cat_list, labels)` reconstructs the original `pd.DataFrame` with original column order, datatypes, categorical string categories, and target column names.

### 2.2 Phase 02: Diffusion Engine
- **Modules:** `src/tabsynth/diffusion/numerical/gaussian_diffusion.py`, `src/tabsynth/diffusion/categorical/categorical_diffusion.py`, and `src/tabsynth/diffusion/mixed/mixed_diffusion.py`.
- **Continuous Gaussian Diffusion:**
  - Forward corruption: $q(\mathbf{x}_t^{\text{num}} \mid \mathbf{x}_0^{\text{num}}) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I}\right)$.
  - Reparameterization target: injected Gaussian noise $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$.
  - Loss: $\mathcal{L}_{\text{num}} = \|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}\|^2_2$ (MSE).
- **Discrete D3PM Categorical Diffusion:**
  - Forward corruption: $q(\mathbf{x}_t^{(j)} \mid \mathbf{x}_0^{(j)}) = \text{Cat}\left(\mathbf{x}_0^{(j)}\bar{\mathbf{Q}}_t^{(j)}\right)$.
  - Transition matrix: $\mathbf{Q}_t = (1 - \beta_t)\mathbf{I} + \frac{\beta_t}{K_j}\mathbf{1}\mathbf{1}^T$.
  - Exact Bayes reverse posterior:
    $$q(\mathbf{x}_{t-1}^{(j)} \mid \mathbf{x}_t^{(j)}, \mathbf{x}_0^{(j)}) = \frac{\mathbf{x}_t^{(j)}\mathbf{Q}_t^{(j)T} \odot \mathbf{x}_0^{(j)}\bar{\mathbf{Q}}_{t-1}^{(j)}}{\mathbf{x}_0^{(j)}\bar{\mathbf{Q}}_t^{(j)}\mathbf{x}_t^{(j)T}}$$
  - Learned reverse transition: marginalizes over predicted clean logits $\hat{\mathbf{z}}^{(j)}$:
    $$p_{\theta}(\mathbf{x}_{t-1}^{(j)} \mid \mathbf{x}_t^{(j)}) = \sum_{k=1}^{K_j} q(\mathbf{x}_{t-1}^{(j)} \mid \mathbf{x}_t^{(j)}, \mathbf{e}_k) \cdot \text{softmax}(\hat{\mathbf{z}}^{(j)})_k$$
  - Loss: $\mathcal{L}_{\text{cat}} = \frac{1}{D_{\text{cat}}} \sum_{j=1}^{D_{\text{cat}}} D_{\text{KL}}\left(q(\mathbf{x}_{t-1}^{(j)} \mid \mathbf{x}_t^{(j)}, \mathbf{x}_0^{(j)}) \parallel p_{\theta}(\mathbf{x}_{t-1}^{(j)} \mid \mathbf{x}_t^{(j)})\right)$.
  - Auxiliary clean cross-entropy: $\mathcal{L}_{\text{aux}} = -\frac{1}{D_{\text{cat}}}\sum_{j=1}^{D_{\text{cat}}} \mathbf{x}_0^{(j)} \log \text{softmax}(\hat{\mathbf{z}}^{(j)})$.

### 2.3 Phase 03: Denoising Neural Architectures
- **Modules:** `src/tabsynth/models/hybrid.py` (`TabSynthHybridDenoiser`), `src/tabsynth/models/mlp_baseline.py` (`TabSynthMLPDenoiser`), and `src/tabsynth/models/integration.py` (`ModelDiffusionIntegrator`).
- **Feature Tokenization:**
  - Maps $D_{\text{num}}$ continuous and $D_{\text{cat}}$ categorical features into sequence of tokens $(B, D, d_{\text{model}})$.
  - Adds learned Column Identity Embeddings $\mathbf{E}_{\text{col}} \in \mathbb{R}^{D \times d_{\text{model}}}$ and Feature Type Embeddings $\mathbf{E}_{\text{type}} \in \mathbb{R}^{2 \times d_{\text{model}}}$.
- **Conditioning & CFG Invariant:**
  - Timestep conditioning: `MixedTimestepEmbedding` fuses continuous sinusoidal embedding and discrete sinusoidal embedding into $(B, d_{\text{model}})$.
  - Class conditioning: `ClassConditioningEmbedding` embeds target class label $Y \in \{0, \dots, C-1\}$. In unconditional mode or during training with dropout probability $p_{\text{uncond}} = 0.15$, label is substituted with null token $\emptyset$ (index $C$).
  - Target $Y$ is strictly conditioning context $c$; it is never an input column to `FeatureTokenizer` or corrupted by diffusion.
- **Model Calling Interface & Container:**
  - Output is returned via `ModelOutput(predicted_noise, categorical_logits)`, supporting dictionary access (`out["predicted_noise"]`), attribute access (`out.predicted_noise`), and tuple unpacking (`pred_eps, pred_logits = model(...)`).
  - Supports both full explicit keyword arguments (`x_num_t`, `x_cat_t`, `timesteps_num`, `timesteps_cat`, `class_labels`) and backward compatibility aliases (`x_cat_t_list`, `t`, `c`).

---

## 3. Compatibility Matrix

| Component | Target File | Status | Compatibility Notes |
| :--- | :--- | :--- | :--- |
| **Optimizer** | `src/tabsynth/training/optimizer.py` | New | AdamW, Adam, SGD with parameter group decay separation (weights decayed, 1D biases/LayerNorms excluded). |
| **Scheduler** | `src/tabsynth/training/scheduler.py` | New | CosineAnnealingLR with linear warmup, LinearLR, StepLR factory. |
| **Loss Function** | `src/tabsynth/training/losses.py` | Update | Standardize multi-objective loss calculation (`TabSynthLoss`) supporting both standalone and `MixedTypeDiffusion` workflows. |
| **Metrics** | `src/tabsynth/training/metrics.py` | New | Running loss accumulator, epoch metrics tracker. |
| **Early Stopping** | `src/tabsynth/training/early_stopping.py` | New | Tracks validation loss with configurable patience, minimum delta, and best model weight restoration. |
| **Checkpointing** | `src/tabsynth/training/checkpoint.py` | Enhance | Rolling top-k checkpoint retention, saving model state dict, optimizer state dict, scheduler state dict, preprocessor, schema, epoch, val_loss, and hyperparameters metadata. |
| **Train Step** | `src/tabsynth/training/train_step.py` | New | Modular single-step forward corruption, neural denoising, loss computation, gradient clipping ($\le 1.0$), and optimizer step. |
| **Validation** | `src/tabsynth/training/validation.py` | New | Standalone validation loop under `torch.no_grad()` computing validation loss components. |
| **Deterministic Seed** | `src/tabsynth/training/seed.py` | New | Deterministic seeding utility for Python `random`, `numpy`, `torch.manual_seed`, and MPS seeds. |
| **Trainer Engine** | `src/tabsynth/training/trainer.py` | Refactor/Enhance | Comprehensive `TabSynthTrainer` integrating all modular components, early stopping, LR scheduling, checkpointing, and MPS memory cache flushing. |
| **CFG Guidance** | `src/tabsynth/sampling/guidance.py` | New | Mathematical extrapolation engine for continuous noise and discrete clean logits. |
| **Conditional Sampler** | `src/tabsynth/sampling/conditional_sampler.py` | New | Class-conditioned sampler generating $P(X \mid Y=c)$ supporting balanced, natural empirical, and class-specific strategies. |
| **Unconditional Sampler** | `src/tabsynth/sampling/unconditional_sampler.py` | New | Generates samples without class conditioning (or using null token). |
| **Mixed Sampler** | `src/tabsynth/sampling/mixed_sampler.py` | New | Joint reverse ancestral sampler for continuous Gaussian and discrete D3PM processes. |
| **Generation Pipeline** | `src/tabsynth/sampling/generation.py` | New | High-level synthesis engine mapping diffusion output tensors to fully formatted `pd.DataFrame` via `preprocessor.inverse_transform`. |
| **Unified Sampler** | `src/tabsynth/sampling/sampler.py` | Refactor | Backward-compatible facade delegating to modular sampling submodules. |

---

## 4. Hardware Optimization Verification (Apple M4)

1. **Native MPS Acceleration:**
   - Metal Performance Shaders (`mps`) is fully functional on macOS Darwin 27.0.1.
   - All diffusion tensors maintain strict `torch.float32` precision.
2. **MPS Memory Cache Management:**
   - At the completion of each epoch and major generation chunk, `torch.mps.empty_cache()` will be invoked to avoid Unified Memory fragmentation and virtual memory pressure.
3. **CPU Fallback:**
   - Graceful fallback to `torch.device("cpu")` if MPS is unavailable or disabled via `--device cpu`.

---

## 5. Backward Compatibility & Test Suite Invariant

All 67 existing tests across `tests/unit/`, `tests/diffusion/`, `tests/models/`, and `tests/integration/` must pass without regression. Any enhancements to `losses.py`, `trainer.py`, `checkpoint.py`, and `sampler.py` will maintain exact API signature compatibility while exposing modular subcomponents for new dedicated test suites in `tests/training/` and `tests/sampling/`.
