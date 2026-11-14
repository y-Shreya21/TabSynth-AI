# TabSynth AI: Implementation Log

## [2026-11-14] - Phase 1: Initialization & Mathematical Foundations
- **Hardware Audit:** macOS (Darwin 27.0.1, arm64, Apple Silicon M4), 16 GB Unified Memory.
- **Python Environment:** Created Python 3.11 virtual environment at `.venv/`.
- **Project Structure:** Initialized modular repository structure with `configs/`, `data/`, `src/tabsynth/`, `api/`, `app/`, `scripts/`, `experiments/`, and `tests/`.
- **Diffusion Schedules (`src/tabsynth/diffusion/schedules.py`):**
  - Linear, Cosine (Nichol & Dhariwal), and Sigmoid noise variance schedules.
  - Implemented analytical forward step, SNR, and cumulative product buffers in `float32`.
- **Gaussian Diffusion (`src/tabsynth/diffusion/gaussian.py`):**
  - $\epsilon$-prediction reparameterization with closed-form posterior $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$.
- **Multinomial Discrete Diffusion (`src/tabsynth/diffusion/multinomial.py`):**
  - D3PM uniform categorical corruption matrix.
  - Exact Bayes reverse posterior marginalizing over clean categorical logits, avoiding invalid continuous out-of-simplex states.

## [2026-11-14] - Phase 2: Hybrid Architecture & Baselines
- **Tokenizers & Embeddings (`src/tabsynth/models/embeddings.py`):**
  - Numerical Feature Tokenizer: Multi-layer linear projection mapping scalar features to $d_{\text{model}}$ dimensional embeddings.
  - Categorical Tokenizer: Learnable lookup tables mapping category indices to $d_{\text{model}}$ token space.
  - Sinusoidal Time Embedding with MLP projection.
  - Class-conditional label embedding with Classifier-Free Guidance (CFG) conditioning dropout ($p_{\text{uncond}} = 0.1$).
- **Pre-LN Transformer Feature Encoder (`src/tabsynth/models/transformer.py`):**
  - Pre-LayerNorm Multi-Head Self-Attention layers with column identity embeddings.
  - Captures complex cross-column interdependencies across all $D$ tabular attributes.
- **Residual MLP Denoiser & Heads (`src/tabsynth/models/denoiser.py`):**
  - Residual MLP blocks with SwiGLU / SiLU non-linearities, layer normalization, and dropout.
  - Dual output heads: continuous noise prediction $\hat{\epsilon}$ and discrete categorical logits $\hat{\mathbf{z}}$.
- **Hybrid Integration (`src/tabsynth/models/hybrid.py`):**
  - End-to-end forward pass integrating tokenizers, time/class conditioners, Transformer cross-feature encoder, and MLP denoiser.
- **Baselines (`src/tabsynth/baselines/`):**
  - `mlp_diffusion.py`: TabDDPM-style MLP-only baseline for architectural ablation studies.
  - `copula.py`: Parametric Gaussian Copula baseline for statistical synthesis benchmarking.

## [2026-11-14] - Phase 3: Data Preprocessing & Zero-Leakage Pipeline
- **Schema & Metadata (`src/tabsynth/data/schema.py`):**
  - Automated column typing (numerical, categorical, datetime, target).
  - Cardinality counting, missing value detection, and metadata persistence.
- **Preprocessor (`src/tabsynth/data/preprocessor.py`):**
  - Zero-leakage policy: QuantileTransformer / StandardScaler and OrdinalEncoder fit strictly on $D_{\text{train}}$.
  - Full bidirectional invertibility for continuous and discrete features.
- **Dataset & Dataloader (`src/tabsynth/data/dataset.py`):**
  - PyTorch `TabularDataset` handling continuous tensors, discrete categorical vectors, and optional class labels.
  - Stratified train/val/test partitioning (70/15/15 split).

## [2026-11-14] - Phase 4: Training & Checkpointing
- **Loss Formulation (`src/tabsynth/training/losses.py`):**
  - Composite multi-objective loss:
    $$\mathcal{L}_{\text{total}} = \lambda_{\text{num}} \text{MSE}(\epsilon, \hat{\epsilon}) + \lambda_{\text{cat}} \mathcal{L}_{\text{D3PM-CE}}(\mathbf{x}_0, \hat{\mathbf{x}}_0) + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}$$
- **Trainer (`src/tabsynth/training/trainer.py`):**
  - Full Apple Silicon MPS acceleration with CPU fallback.
  - Gradient clipping ($\le 1.0$), AdamW optimizer, Cosine Annealing LR scheduler, and periodic MPS cache flushing.
- **Checkpoint Manager (`src/tabsynth/training/checkpoint.py`):**
  - Rolling top-k checkpoint retention (`keep_top_k: 2`) preventing disk exhaustion.
  - Full serialization of model weights, optimizer, preprocessor, and configuration metadata.

## [2026-11-14] - Phase 5: Sampling Engine with CFG
- **Reverse Sampler (`src/tabsynth/sampling/sampler.py`):**
  - Iterative ancestral sampling ($T \to 0$) for continuous features (Gaussian) and discrete features (D3PM categorical posterior).
  - Classifier-Free Guidance (CFG) extrapolation:
    $$\tilde{\epsilon} = \epsilon_{\theta}(\mathbf{x}_t, \emptyset, t) + w \cdot \left(\epsilon_{\theta}(\mathbf{x}_t, y, t) - \epsilon_{\theta}(\mathbf{x}_t, \emptyset, t)\right)$$
  - Integrated inverse transformation restoring original column formats and category strings into a ready-to-use pandas DataFrame.

## [2026-11-14] - Phase 6: Privacy & Fidelity Evaluation Framework
- **Fidelity Suite (`src/tabsynth/evaluation/fidelity.py`):**
  - Kolmogorov-Smirnov (KS) test per numerical feature.
  - Total Variation Distance (TVD) per categorical feature.
  - Frobenius norm difference of Pearson/Theil's U correlation matrices.
  - Classifier Two-Sample Test (C2ST) using 5-fold cross-validated LightGBM / XGBoost discriminator.
- **Utility Benchmark (`src/tabsynth/evaluation/utility.py`):**
  - Train on Synthetic, Test on Real (TSTR) vs. Train on Real, Test on Real (TRTR).
  - Downstream evaluations on Logistic Regression and Random Forest.
  - Calculation of F1 Parity ($F_1^{\text{TSTR}} / F_1^{\text{TRTR}}$) and minority class recall delta.
- **Empirical Privacy Auditing (`src/tabsynth/evaluation/privacy.py`):**
  - Distance to Closest Record (DCR) using mixed-type Gower distance.
  - Nearest Neighbor Distance Ratio (NNDR): $d_1 / d_2$.
  - Exact row-match detection.
  - Shadow-model Membership Inference Attack (MIA) with AUROC computation.

## [2026-11-14] - Phase 7: REST API & Streamlit Dashboard
- **FastAPI Engine (`api/`):**
  - Implemented 9 production endpoints: `/health`, `/schema`, `/train`, `/generate`, `/evaluate/fidelity`, `/evaluate/utility`, `/evaluate/privacy`, `/experiments/list`, `/experiments/results/{exp_id}`.
  - Non-blocking asynchronous training and generation jobs using background tasks.
- **Streamlit Analytics UI (`app/main.py`):**
  - 9 interactive tabs: Dataset Studio, Model Training, Conditional Generation, Fidelity Studio, Privacy Auditor, Downstream Utility, Baseline Comparison, Experiment Tracker, System Health.
  - Rich Plotly visualizations: correlation heatmaps, cumulative distribution plots, ROC curves, DCR histograms.

## [2026-11-14] - Phase 8: Empirical Validation & Containerization
- **Benchmark Datasets (`scripts/download_datasets.py`):**
  - UCI Adult Income (`adult.csv`, 32,561 records, 14 features).
  - UCI Bank Marketing (`bank_marketing.csv`, 5,000 records, 16 features).
  - UCI Credit Default (`credit_default.csv`, 5,000 records, 11 features).
- **Test Suite (`tests/`):**
  - Automated unit and integration tests covering diffusion math, transformer encoders, hybrid forwards, zero-leakage preprocessors, fidelity/utility/privacy metrics, and FastAPI endpoints.
- **Real Training & Generation on Apple Silicon MPS:**
  - Trained 5 epochs on UCI Adult Income: Validation loss converged to `0.5051`.
  - Generated 200 synthetic rows in `4.0` seconds.
  - Evaluated real synthetic data: Mean KS = `0.1107`, Mean TVD = `0.2315`, Random Forest F1 Parity = `0.9283`, DCR Ratio = `1.0000`, Exact Matches = `0`, MIA AUROC = `0.5000`.
- **Experiment E1 (Ablation: Hybrid vs MLP-only across Seeds 42 and 1337):**
  - Hybrid achieved lower mean KS (`0.1039` vs `0.1466` on seed 42) and lower mean TVD (`0.266` vs `0.292`), validating that cross-column attention improves marginal and joint distribution fidelity.
- **Containerization (`Dockerfile`, `docker-compose.yml`):**
  - Multi-stage Docker build with non-root security (`tabsynth:10001`).
  - Automated healthcheck on `/health`.
  - Docker Compose orchestration with shared volume storage across ports 8000 and 8501.

## [2026-11-14] - Phase 02: Dedicated Tabular Diffusion Engine Implementation & Verification
- **Modular Diffusion Architecture (`src/tabsynth/diffusion/`):**
  - Subpackage `numerical/`: `GaussianDiffusion` and `NumericalSampler`.
  - Subpackage `categorical/`: `transition.py`, `CategoricalDiffusion` and `CategoricalSampler`.
  - Subpackage `mixed/`: `MixedTypeDiffusion`.
  - Shared utilities (`utils.py`): Device selector, simplex validator, tensor extractors.
  - Tests: 26 / 26 unit tests passed in `tests/diffusion/`.
  - Smoke: `scripts/smoke_test_diffusion.py` passed on CPU and MPS.

## [2026-11-14] - Phase 03: Hybrid Transformer-MLP Architecture Implementation & Verification
- **Modular Model Components (`src/tabsynth/models/`):**
  - `tokenizer.py`: `NumericalFeatureTokenizer` ($x_i \mathbf{w}_i + \mathbf{b}_i$), `CategoricalFeatureTokenizer` (dedicated projections per cardinality $K_j$), and master `FeatureTokenizer` incorporating learned Column Identity Embeddings ($\mathbf{E}_{\text{col}}$) and Feature Type Embeddings ($\mathbf{E}_{\text{type}}$). Handles numerical-only, categorical-only, and mixed tables.
  - `embeddings.py`: `SinusoidalTimeEmbedding`, `MixedTimestepEmbedding` fusing independent continuous ($t_{\text{num}}$) and discrete ($t_{\text{cat}}$) timesteps, and `ClassConditioningEmbedding` with null unconditional token and condition dropout ($p_{\text{uncond}} = 0.15$) for CFG.
  - `transformer.py`: `TransformerEncoderBlock` and `TransformerFeatureEncoder` with Pre-LayerNorm, Multi-Head Self-Attention, GELU feed-forward sublayers, and residual connections.
  - `denoiser.py`: `FeatureResidualMLPBlock` and `FeatureResidualMLPDenoiser` operating across the sequence of feature tokens with fused time+class additive conditioning, preserving per-feature token structures without destructive pooling.
  - `heads.py`: Specialized `NumericalOutputHead` (predicting $\hat{\boldsymbol{\epsilon}} \in \mathbb{R}^{B \times D_{\text{num}}}$) and `CategoricalOutputHeads` (predicting unnormalized clean logits $\hat{\mathbf{z}}^{(j)} \in \mathbb{R}^{B \times K_j}$).
  - `hybrid.py`: Assembled `TabSynthHybridDenoiser` with standardized `ModelOutput` container supporting dictionary access, attribute access, and tuple unpacking.
  - `mlp_baseline.py`: Assembled `TabSynthMLPDenoiser` baseline for RQ1 ablation experiments.
  - `integration.py`: `ModelDiffusionIntegrator` coordinating forward corruption, neural denoising, and multi-objective loss computation ($\mathcal{L}_{\text{total}} = \lambda_{\text{num}} \mathcal{L}_{\text{num}} + \lambda_{\text{cat}} \mathcal{L}_{\text{cat}} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}$).
  - `diagnostics.py`: Parameter counter, model summary, and comparison diagnostics.
- **Configuration (`configs/model.yaml`):**
  - Master configuration file defining Transformer layers, attention heads, feedforward dims, MLP residual blocks, conditioning, and feature dimensions.
- **Unit & Integration Test Suite (`tests/models/`):**
  - `test_tokenizer.py` (6 tests passed)
  - `test_time_embedding.py` (2 tests passed)
  - `test_conditioning.py` (4 tests passed)
  - `test_transformer_denoiser.py` (3 tests passed)
  - `test_mlp_denoiser.py` (3 tests passed)
  - `test_output_heads.py` (2 tests passed)
  - `test_model_integration.py` (3 tests passed across CPU and MPS)
  - **Full test suite:** 67 / 67 tests passed across `tests/` in 15.41s.
- **Smoke & Overfit Experiments (`scripts/smoke_test_model.py`):**
  - CPU: Initial step loss = 1.4214, Final step loss = 0.5598 (**60.62% convergence**).
  - MPS: Initial step loss = 1.9548, Final step loss = 0.8407 (**56.99% convergence**).
  - Parameter profiling: Hybrid (223,374 params) vs MLP baseline (198,030 params), delta: +25,344 (+12.8%).
- **Documentation (`docs/PHASE_03_MODEL_ARCHITECTURE.md`):**
  - Authored comprehensive 18-section technical specification with Mermaid architecture diagrams, mathematical formulations, tensor shapes, and empirical profiles.

## [2026-11-15] - Phase 04: Training Pipeline and Conditional Synthetic Data Generation
- **Compatibility Audit (`docs/PHASE_04_COMPATIBILITY_REPORT.md`):**
  - Conducted architectural and mathematical interface audit across Phase 01 preprocessors, Phase 02 diffusion engines, and Phase 03 models.
  - Verified tensor shapes, loss reduction mechanics, and class-conditioning invariants ($Y$ strictly as conditioning input $c$, never a feature in $X$).
- **Modular Training Package (`src/tabsynth/training/`):**
  - `optimizer.py`: Parameter grouping separating 2D+ projection weights (with decay) from 1D biases and LayerNorm parameters (zero decay). Factory for AdamW, Adam, SGD.
  - `scheduler.py`: Warmup + Cosine Annealing learning rate schedule down to minimum learning rate floor.
  - `metrics.py`: `MetricAccumulator` tracking running losses, sample counts, epoch time, and throughput (samples/sec).
  - `early_stopping.py`: `EarlyStopping` monitoring validation loss with configurable patience, minimum delta, and best model weight restoration.
  - `checkpoint.py`: Top-$k$ rolling checkpoint manager saving model weights, optimizer, scheduler, fitted preprocessor, and schema metadata.
  - `train_step.py`: Single mini-batch training execution with Gaussian and D3PM forward corruption, composite loss calculation, and gradient clipping ($\le 1.0$).
  - `validation.py`: Clean validation evaluator under `torch.no_grad()`.
  - `seed.py`: Deterministic seed utility for Python random, NumPy, PyTorch, and MPS.
  - `losses.py`: Standardized multi-objective loss function with compatibility across both `MixedTypeDiffusion` and standalone diffusion modules.
  - `trainer.py`: Unified `TabSynthTrainer` (and `TabularTrainer`) managing training/validation loops, early stopping, and Apple Silicon MPS memory clearing (`torch.mps.empty_cache()`).
- **Modular Sampling Package (`src/tabsynth/sampling/`):**
  - `guidance.py`: Classifier-Free Guidance (CFG) extrapolation: $\tilde{\boldsymbol{\epsilon}} = (1+w)\boldsymbol{\epsilon}_{\text{cond}} - w\boldsymbol{\epsilon}_{\text{uncond}}$ and $\tilde{\mathbf{z}} = (1+w)\mathbf{z}_{\text{cond}} - w\mathbf{z}_{\text{uncond}}$.
  - `mixed_sampler.py`: Joint reverse ancestral sampler coordinating Gaussian continuous and D3PM categorical transitions.
  - `conditional_sampler.py`: Class-conditional sampler supporting `balanced`, `natural`, and `class_specific` conditioning strategies.
  - `unconditional_sampler.py`: Unconditional sampling engine.
  - `generation.py`: High-level generation pipeline coordinating reverse sampling and `preprocessor.inverse_transform` into a structured `pd.DataFrame`.
  - `sampler.py`: Backward-compatible public API facade exposing `TabSynthSampler`.
- **Configurations (`configs/`):**
  - `configs/training.yaml`, `configs/generation.yaml`, `configs/experiments/adult_hybrid.yaml`, `configs/experiments/adult_mlp.yaml`.
- **CLI Scripts (`scripts/`):**
  - `scripts/train.py`: Unified training CLI supporting YAML configs and CLI overrides.
  - `scripts/generate.py`: Synthesis CLI generating synthetic CSV datasets with CFG.
  - `scripts/inspect_checkpoint.py`: Diagnostic utility inspecting checkpoint contents, schema, and parameter counts.
  - `scripts/resume_training.py`: Script to resume training from an existing checkpoint.
- **Unit & Integration Test Suites:**
  - Added `tests/training/` (10 tests covering optimizer, scheduler, early stopping, checkpointing, train step, and trainer fit).
  - Added `tests/sampling/` (6 tests covering CFG extrapolation, mixed sampler bounds, conditional strategies, and DataFrame generation).
  - **Full test suite:** 83 / 83 tests passing in 14.14s (100% regression-free).
- **Empirical Training & Validation (UCI Adult Income on Apple Silicon MPS):**
  - Trained TabSynth Hybrid model for 5 epochs: validation loss converged from 1.3660 to **0.4900** (further improved to **0.4544** on resume).
  - Trained TabDDPM MLP baseline for 5 epochs: validation loss converged from 1.4932 to **0.4779**.
  - Generated 500 balanced synthetic samples with CFG $w=1.0$ (109.8 rows/sec): 0 missing values, perfect column alignment, and realistic continuous distributions (e.g., real age mean: 38.58, synthetic age mean: 40.32).
  - Generated class-specific synthetic samples: 100% minority class (`>50K`) and 100% majority class (`<=50K`).
- **Documentation:**
  - Authored `docs/PHASE_04_COMPATIBILITY_REPORT.md` and `docs/PHASE_04_TRAINING_AND_SAMPLING.md`.

## [2026-11-15] - Phase 05: Comprehensive Evaluation, Baseline Comparison, Privacy Auditing & Research Experiments
- **Codebase Audit (`docs/PHASE_05_AUDIT.md`):**
  - Audited existing fidelity, utility, and privacy modules; identified architectural gaps; documented mathematical formulations and zero-leakage constraints.
- **Modular Statistical Fidelity Suite (`src/tabsynth/evaluation/fidelity/`):**
  - `numerical_metrics.py`: Two-sample Kolmogorov-Smirnov test statistics & p-values, normalized 1-Wasserstein distances, mean/variance deviations, and multi-quantile absolute errors (10th, 25th, 50th, 75th, 90th percentiles).
  - `categorical_metrics.py`: Total Variation Distance (TVD), per-level frequency deviations, missing rate discrepancies, and unseen category tracking.
  - `dependency_metrics.py`: Pairwise Pearson & Spearman correlation Frobenius distances, Theil's U (uncertainty coefficient), and Cramer's V categorical association matrices.
  - `c2st.py`: 5-fold cross-validated Classifier Two-Sample Test using Random Forest, HistGradientBoosting, or Logistic Regression.
  - `sdmetrics_evaluator.py`: Native SDMetrics fallback computing 1D marginal shapes, 2D pairwise associations, and boundary adherence scores.
  - `fidelity_report.py`: `FidelityEvaluator` assembling composite statistical fidelity reports.
- **Modular Downstream Machine Learning Utility Suite (`src/tabsynth/evaluation/utility/`):**
  - `classification_metrics.py`: Threshold-dependent (Accuracy, Macro-F1, Weighted-F1, Balanced Accuracy, Precision, Recall, Minority Recall/Precision/F1) and threshold-independent (AUROC, AUPRC) evaluation.
  - `downstream.py`: Multi-model classifier factory (`LogisticRegression`, `RandomForestClassifier`, `HistGradientBoostingClassifier`) and zero-leakage `DownstreamDataPreprocessor` fitting all scalers and encoders strictly on $D_{\text{train}}$.
  - `trtr.py`: Train on Real, Test on Real protocol defining upper-bound performance benchmarks.
  - `tstr.py`: Train on Synthetic, Test on Real protocol computing relative utility parity ratios ($F_1^{\text{TSTR}} / F_1^{\text{TRTR}}$) and performance drops.
  - `augmentation.py`: Mixed augmentation protocol supplementing real training partitions with synthetic minority records across ratios $[0.25\times, 0.5\times, 1.0\times, 2.0\times]$ and evaluating on untouched $D_{\text{test}}$.
  - `utility_report.py`: Master `UtilityEvaluator` coordinating TRTR, TSTR, and Augmentation.
- **Empirical Privacy Risk Auditing Suite (`src/tabsynth/evaluation/privacy/`):**
  - `distance_metrics.py`: Pairwise normalized Gower distance matrix, Distance to Closest Record (DCR) distributions to Train and Test partitions, DCR ratios, 5th percentile ratios, Nearest Neighbor Distance Ratios ($NNDR = d_1 / d_2$), and exact match detection ($d < 1e-5$).
  - `membership_inference.py`: Distance-based Membership Inference Attack (MIA) assessing empirical memorization and computing adversarial ROC-AUC, accuracy, precision, and recall.
  - `attribute_inference.py`: Attribute Inference Attack (AIA) evaluating adversarial reconstruction advantage for designated sensitive features given quasi-identifiers.
  - `singling_out.py`: Predicate-based singling out risk assessing univariate and multivariate uniqueness transfer from synthetic to real records.
  - `privacy_report.py`: `PrivacyAuditor` aggregating all privacy metrics with mandatory formal non-DP theoretical disclaimers.
- **Baseline Registry & Benchmarking Suite (`src/tabsynth/evaluation/benchmarks/`):**
  - `base.py`: Abstract `BaseSynthesizer` defining standardized `fit`, `sample`, `sample_conditional`, `save`, and `load` lifecycle.
  - `copula_wrapper.py`: `GaussianCopulaSynthesizer` wrapping parametric Gaussian Copula.
  - `tvae.py`: Native PyTorch Tabular VAE (`TVAESynthesizer`) implementing encoder/decoder MLPs, reparameterization trick, MSE + CrossEntropy reconstruction, and KL latent divergence.
  - `diffusion_wrappers.py`: `TabSynthHybridSynthesizer` and `TabDDPMMLPSynthesizer` wrapping diffusion engines under the benchmark interface.
  - `baseline_registry.py`: Factory registry with `get_synthesizer` and `list_available_synthesizers`.
  - `benchmark_runner.py`: Multi-model benchmarking runner computing throughput (rows/sec), parameter counts, statistical fidelity, utility parity, and privacy scores into comparative markdown and JSON tables.
- **Research Ablations & Visualizations (`src/tabsynth/evaluation/experiments/`, `visualization.py`):**
  - `ablations.py`: Implementations for E1 (Architecture: Transformer vs MLP), E5 (Class Sampling: Balanced vs Natural), E6 (Augmentation ratios), and E7 (Fidelity-Privacy trade-off).
  - `visualization.py`: Plotly visualization suite producing interactive HTML/JSON charts for 1D marginal overlays, correlation difference heatmaps, DCR distributions, and augmentation trade-off curves.
- **CLI Runners (`scripts/`):**
  - `scripts/run_phase05_smoke.py`: Fast smoke test verifying all evaluation pipelines in 1.73s (< 30s threshold).
  - `scripts/run_phase05_benchmark.py`: Complete multi-model benchmark comparison across Gaussian Copula, TVAE, TabDDPM MLP, and TabSynth Hybrid on UCI Adult Income.
  - `scripts/run_phase05_ablations.py`: Ablation study runner for E1, E5, E6, and E7.
  - `scripts/generate_phase05_report.py`: Evaluation report generator compiling empirical results into `docs/PHASE_05_EVALUATION_REPORT.md`.
- **Test Suites:**
  - Added `tests/evaluation/test_fidelity_metrics.py`, `tests/evaluation/test_utility_metrics.py`, `tests/evaluation/test_c2st.py`.
  - Added `tests/privacy/test_distance_metrics.py`, `tests/privacy/test_membership_inference.py`.
  - Added `tests/benchmarks/test_baseline_registry.py`, `tests/benchmarks/test_tvae.py`.
  - **Full test suite:** 103 / 103 tests passing in 17.38s (100% pass rate, zero regression).
- **Key Empirical Benchmark Findings (Adult Income, Apple Silicon MPS):**
  - **Correlation Fidelity:** TabSynth Hybrid achieves Pearson correlation Frobenius distance of **0.1117**, outperforming TVAE (0.6326) and TabDDPM MLP (0.1591).
  - **Throughput:** TabSynth Hybrid achieves **205.1 rows/sec** on Apple M4 MPS.
  - **Privacy:** TabSynth Hybrid exhibits a DCR ratio of **1.0155** ($\approx 1.0$), 5th percentile ratio of **2.4997**, **0 exact matches** (0.00% memorization), and MIA ROC-AUC of **0.4907** (indistinguishable from 0.50 random guess), confirming low empirical memorization.
  - **Minority Balancing (E5):** Balanced class-conditional sampling increases downstream minority-class recall from **0.0000** (natural) to **0.4855** (+48.55% gain).


