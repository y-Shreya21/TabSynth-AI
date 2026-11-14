# TabSynth AI: Phase 05 Repository & Architecture Audit Report

**Date:** 2026-11-15  
**Hardware Environment:** Apple Silicon M4 (macOS Darwin 27.0.1, arm64, 16 GB Unified Memory)  
**Execution Runtime:** Python 3.11.15, PyTorch 2.15.0 with native Apple MPS (`torch.device("mps")`) and CPU fallback  
**Status:** Audit Completed – Ready for Phase 05 Implementation  

---

## 1. Executive Summary

This audit reviews the entire TabSynth AI codebase across Phases 01 through 04 to establish the foundation for **Phase 05: Comprehensive Evaluation, Baseline Comparison, Privacy Auditing & Research Experiments**.

The primary objective of Phase 05 is to implement an objective, rigorous, and reproducible evaluation framework determining whether TabSynth AI:
1. Preserves the marginal and joint statistical distributions of real tabular datasets.
2. Captures complex non-linear feature relationships and categorical dependencies.
3. Produces synthetic records effective for downstream machine learning (TRTR, TSTR, Mixed Augmentation).
4. Enhances minority-class predictive performance when used for targeted conditional augmentation.
5. Exhibits acceptable empirical privacy-risk profiles without catastrophic training memorization.
6. Demonstrates clear, measurable trade-offs against established synthetic data baselines.

In accordance with Phase 05 guidelines: **No results, metrics, or comparisons will ever be fabricated.** All measurements are produced from real execution against empirical data partitions.

---

## 2. Review of Existing Implementations (Phases 01–04)

### 2.1 Phase 01: Preprocessing & Data Layer
- **Modules:** [`src/tabsynth/data/schema.py`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/data/schema.py), [`src/tabsynth/data/preprocessor.py`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/data/preprocessor.py), [`src/tabsynth/data/dataset.py`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/data/dataset.py).
- **Zero-Leakage Policy:** Preprocessor is fitted **strictly on the 70% training split** ($D_{\text{train}}$). Scalers, quantile transformers, and categorical vocabularies never access validation ($D_{\text{val}}$) or test ($D_{\text{test}}$) sets.
- **Datasets Available:**
  1. `data/raw/adult.csv`: 32,561 rows, 15 columns (5 continuous, 9 categorical, binary target `income`).
  2. `data/raw/bank_marketing.csv`: 5,000 rows, 17 columns (7 continuous, 9 categorical, binary target `y`).
  3. `data/raw/credit_default.csv`: 5,000 rows, 12 columns (9 continuous, 2 categorical, binary target `default.payment.next.month`).
- **Data Tensor Formats:**
  - `x_num`: $\mathbb{R}^{B \times D_{\text{num}}}$ continuous features (`float32`), bounded in $[-5.0, 5.0]$.
  - `x_cat_list`: List of $D_{\text{cat}}$ one-hot tensors, each of shape $\mathbb{R}^{B \times K_j}$ (`float32`).
  - `labels`: $\mathbb{Z}^{B}$ target class indices (`int64`), strictly separated from predictor features.

### 2.2 Phase 02: Diffusion Engine
- **Modules:** [`src/tabsynth/diffusion/numerical/`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/diffusion/numerical/), [`src/tabsynth/diffusion/categorical/`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/diffusion/categorical/), [`src/tabsynth/diffusion/mixed/`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/diffusion/mixed/).
- **Mathematical Formulations:**
  - Gaussian continuous diffusion with cosine beta schedule, $\epsilon$-prediction objective $\mathcal{L}_{\text{num}} = \|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}\|^2_2$.
  - Discrete D3PM categorical transition matrices with exact Bayes reverse posterior, probability marginalization over clean logits, variational KL objective $\mathcal{L}_{\text{cat}}$, and auxiliary clean cross-entropy $\mathcal{L}_{\text{aux}}$.
  - Unified `MixedTypeDiffusion` orchestrating joint forward corruption and reverse ancestral transitions.

### 2.3 Phase 03: Denoising Neural Architectures
- **Modules:** [`src/tabsynth/models/tokenizer.py`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/models/tokenizer.py), [`src/tabsynth/models/transformer.py`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/models/transformer.py), [`src/tabsynth/models/denoiser.py`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/models/denoiser.py), [`src/tabsynth/models/hybrid.py`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/models/hybrid.py), [`src/tabsynth/models/mlp_baseline.py`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/models/mlp_baseline.py).
- **Architectural Duality:**
  - Proposed `TabSynthHybridDenoiser`: Pre-LN Multi-Head Self-Attention over feature tokens + Feature-level Residual MLP denoiser (240,276 parameters on Adult Income).
  - Baseline `TabSynthMLPDenoiser`: Pure Feature-level Residual MLP denoiser omitting self-attention (177,478 parameters on Adult Income).
- **Conditioning Mechanism:**
  - Target label $Y$ is embedded purely via `ClassConditioningEmbedding` as conditioning context $c$ with dropout probability $p_{\text{uncond}} = 0.15$ for Classifier-Free Guidance (CFG).

### 2.4 Phase 04: Training & Sampling Pipelines
- **Modules:** [`src/tabsynth/training/`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/training/) and [`src/tabsynth/sampling/`](file:///Users/shreyayadav/TabSynth%20AI/src/tabsynth/sampling/).
- **Training Infrastructure:** `TabSynthTrainer`, `execute_train_step`, `evaluate_validation`, `EarlyStopping`, `CheckpointManager`, `build_optimizer` (weight decay parameter grouping), `build_scheduler` (warmup + cosine annealing), and Apple MPS memory clearing (`torch.mps.empty_cache()`).
- **Sampling Pipeline:** `GenerationPipeline`, `MixedAncestralSampler`, `ConditionalSampler` (balanced, natural, class-specific), and `guidance.py` (CFG extrapolation for continuous $\tilde{\boldsymbol{\epsilon}}$ and discrete $\tilde{\mathbf{z}}$).

---

## 3. Available Model Checkpoints

The repository contains pre-trained, validated checkpoints saved in `checkpoints/`:

| Checkpoint Path | Architecture | Dataset | Epoch | Val Loss | Size | Parameters |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `checkpoints/adult_hybrid_best.pt` | `TabSynthHybridDenoiser` | Adult Income | 5 (resumed 6) | **0.4544** | 3.0 MB | 240,276 |
| `checkpoints/adult_mlp_best.pt` | `TabSynthMLPDenoiser` | Adult Income | 5 | **0.4779** | 2.2 MB | 177,478 |

Both checkpoints store the complete bundle: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, fitted `preprocessor`, `schema`, `config`, and `metadata`.

---

## 4. Existing Evaluation Utilities Audit

The repository contains initial evaluation files created in Phase 01:
- `src/tabsynth/evaluation/fidelity.py`: Implements basic Kolmogorov-Smirnov test, Wasserstein distance, Total Variation Distance (TVD), correlation matrix Frobenius norm, and preliminary C2ST with Random Forest.
- `src/tabsynth/evaluation/utility.py`: Implements initial TRTR vs TSTR data preparation with dummy encoding and evaluation with Logistic Regression and Random Forest.
- `src/tabsynth/evaluation/privacy.py`: Implements pairwise Gower distance matrix, Distance to Closest Record (DCR), Nearest Neighbor Distance Ratio (NNDR), and basic distance-based shadow MIA.

### Gaps & Limitations in Existing Evaluation Utilities:
1. **Fidelity:**
   - Numerical metrics lack feature-level mean difference, std difference, and quantile deviation breakdowns.
   - Categorical metrics do not explicitly account for unseen categories or missing frequency differences.
   - Correlation computation converts all categoricals to one-hot dummies and computes Pearson correlation, rather than using statistically rigorous mixed-type associations (Cramer's V, Theil's U, Correlation Ratio).
   - C2ST lacks cross-validated probability calibration, ROC curves, and multi-model discriminators.
   - SDMetrics integration is missing.
2. **Utility:**
   - Mixed augmentation (Real + Synthetic) across configurable ratios (0.25x, 0.5x, 1.0x, 2.0x) is not modularized.
   - Classification metrics lack Average Precision (AUPRC), per-class precision/recall curves, and confusion matrix storage.
   - Gradient Boosting (HistGradientBoosting) is not yet integrated.
3. **Privacy:**
   - Attribute inference attacks are not implemented.
   - Singling-out risk is not implemented.
   - Clear formal warnings distinguishing empirical distance heuristics from differential privacy guarantees are not structured into reporting.
4. **Baselines & Experiments:**
   - No unified baseline registry (`fit()`, `sample()`, `sample_conditional()`, `save()`, `load()`).
   - TVAE baseline is not implemented.
   - No structured experiment runner supporting smoke tests, single-dataset benchmarks, and ablations (E1 through E9).
   - No automated Markdown / table / plot generation pipeline.

---

## 5. Compatibility & Environment Assessment

### 5.1 Local Hardware & Compute Constraints
- **Platform:** Apple Silicon M4 (16 GB Unified Memory).
- **Available Disk Space:** ~2.6 GB on local volume. Checkpointing must strictly adhere to `keep_top_k: 2`, and synthetic batch generation must be chunked to avoid large memory footprints.
- **Network Isolation:** Standard terminal sandbox is network-isolated; packages from PyPI cannot be fetched without explicit user bypass approval.
- **Installed Packages:** `torch` 2.15.0 (with MPS), `scipy` 1.17.1, `scikit-learn` 1.9.1, `pandas` 3.1.1, `numpy` 2.4.6, `plotly` 7.1.0, `pytest` 9.2.0.
- **Uninstalled Optional Packages:** `sdv`, `ctgan`, `sdmetrics`, `anonymeter`, `opacus`, `xgboost`, `lightgbm`.
- **Strategy:**
  - Build standalone, mathematically exact native implementations for TVAE, Gaussian Copula, TabDDPM MLP, and TabSynth Hybrid.
  - Implement full native fidelity, utility, and privacy metrics without external dependencies.
  - Integrate SDMetrics, CTGAN, and Anonymeter with robust fallback / classification if the external library is absent.
  - Use `scikit-learn`'s native `HistGradientBoostingClassifier`, `RandomForestClassifier`, and `LogisticRegression` for high-performance downstream evaluation and C2ST.
  - Use `plotly` for generating self-contained interactive HTML and JSON visualizations.

### 5.2 Differential Privacy (DP-SGD & Opacus) Feasibility
- **Assessment:** DP-SGD requires per-sample gradient computation and clipping ($\|\mathbf{g}_i\|_2 \le C$).
- In TabSynth AI's mixed architecture, the discrete D3PM loss requires marginalization over categorical transition matrices across multiple one-hot states. Per-sample gradient clipping across Transformer self-attention and dynamic multinomial heads on Apple MPS introduces severe memory expansion ($\mathcal{O}(B \cdot D \cdot d_{\text{model}})$ gradient buffers), which would crash within local memory budgets without specialized CUDA Opacus hooks.
- **Resolution:** In accordance with Section 9 of the prompt, the DP-SGD experiment will be rigorously documented as technically investigated and deferred to future CUDA cluster work, with all mathematical and memory constraints fully explained in the final evaluation report.

---

## 6. Proposed Phase 05 Implementation Plan

1. **Modular Fidelity Package (`src/tabsynth/evaluation/fidelity/`):**
   - `numerical_metrics.py`: KS-test, Wasserstein, mean/std diffs, quantile diffs.
   - `categorical_metrics.py`: TVD, frequency error, unseen category tracking.
   - `dependency_metrics.py`: Pearson/Spearman, Theil's U, Cramer's V, Frobenius matrix distance.
   - `c2st.py`: 5-fold cross-validated C2ST with Random Forest & HistGradientBoosting, ROC-AUC, accuracy.
   - `sdmetrics_evaluator.py`: SDMetrics wrapper with native fallback.
   - `fidelity_report.py`: Aggregated fidelity summary.
2. **Modular Utility Package (`src/tabsynth/evaluation/utility/`):**
   - `classification_metrics.py`: Accuracy, macro/weighted F1, balanced accuracy, ROC-AUC, AUPRC, confusion matrix.
   - `trtr.py`: Train on Real, Test on Real protocol.
   - `tstr.py`: Train on Synthetic, Test on Real protocol.
   - `augmentation.py`: Mixed augmentation across ratios [0.25x, 0.5x, 1x, 2x].
   - `downstream.py`: Unified downstream evaluator (Logistic Regression, Random Forest, HistGradientBoosting).
3. **Modular Privacy Package (`src/tabsynth/evaluation/privacy/`):**
   - `distance_metrics.py`: Gower DCR, NNDR, 5th percentile ratios, synthetic-to-train vs synthetic-to-test distributions.
   - `membership_inference.py`: Distance-based MIA and shadow model attack.
   - `attribute_inference.py`: Attribute inference risk evaluation.
   - `singling_out.py`: Singling-out risk assessment.
   - `privacy_report.py`: Aggregated privacy report with formal privacy disclaimers.
4. **Baseline Models & Registry (`src/tabsynth/evaluation/benchmarks/`):**
   - `baseline_registry.py`: Common interface `fit()`, `sample()`, `sample_conditional()`, `save()`, `load()`.
   - `tvae.py`: Native PyTorch Tabular VAE (Encoder, Decoder, Gumbel/Categorical CE, Gaussian continuous, KL).
   - `copula.py`: Native Gaussian Copula synthesizer.
   - `tabddpm.py`: TabDDPM MLP baseline.
   - `benchmark_runner.py`: Orchestrates multi-model benchmark comparison.
   - `comparison.py`: Comparative tables and metric aggregations.
5. **Research Experiments & Ablations (`src/tabsynth/evaluation/experiments/`):**
   - `experiment_runner.py`: Configurable execution modes (`smoke`, `single_dataset`, `benchmark`, `ablations`).
   - `ablations.py`: Ablations E1 (architecture), E4 (conditioning), E5 (class sampling), E6 (augmentation ratios), E7 (fidelity-privacy trade-off), E9 (computational efficiency).
6. **Visualization & Reporting (`src/tabsynth/evaluation/visualization.py`):**
   - Plotly-based visualizers for marginals, correlation heatmaps, C2ST ROC curves, DCR distributions, and trade-off frontiers.
   - Report generator creating `docs/PHASE_05_EVALUATION_REPORT.md`.
7. **CLI Scripts (`scripts/`):**
   - `scripts/run_phase05_smoke.py`: Fast smoke evaluation.
   - `scripts/run_phase05_benchmark.py`: Complete baseline benchmark.
   - `scripts/run_phase05_ablations.py`: Ablation study runner.
   - `scripts/generate_phase05_report.py`: Markdown report generator.
8. **Unit & Integration Test Suites:**
   - Dedicated tests in `tests/evaluation/`, `tests/privacy/`, `tests/benchmarks/`.
   - Ensure all 83 existing tests continue to pass.
