# TabSynth AI: Implementation Plan & Progress Tracking

## Project Objective
Implement **TabSynth AI**: a Hybrid Transformer-MLP continuous-discrete diffusion framework for high-fidelity, privacy-aware, and class-conditional synthetic tabular data generation, optimized for local Apple Silicon M4 with PyTorch MPS.

---

## System Inspection & Development Constraints
- **Hardware Profile:** Apple Silicon M4, macOS (Darwin 27.0.1, arm64).
- **Compute Acceleration:** Apple Metal Performance Shaders (`mps`) with graceful CPU fallback.
- **Unified Memory:** 16 GB unified RAM.
- **Disk Storage:** ~3.0 GB free disk space (monitoring disk space during dataset & checkpoint generation).
- **Python Runtime:** Python 3.11 (`/opt/homebrew/bin/python3.11` creating `.venv`).
- **Precision:** `torch.float32` enforced across all diffusion equations to avoid underflow in cumulative variances.

---

## Implementation Roadmap & Milestone Checklist

### Phase 1: Environment, Inspection & Data Engineering Pipeline
- [x] Inspect local machine, macOS version, Python 3.11, RAM, and disk storage.
- [x] Set up Python 3.11 virtual environment (`.venv`).
- [x] Create project tracking files: `IMPLEMENTATION_PLAN.md`, `IMPLEMENTATION_LOG.md`, `KNOWN_ISSUES.md`.
- [x] Install core dependencies (`torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `pyyaml`, `pytest`, `fastapi`, `uvicorn`, `streamlit`, `plotly`, `httpx`).
- [x] Verify PyTorch MPS acceleration on Apple Silicon M4 (`torch.backends.mps.is_available() = True`).
- [x] Implement dataset downloader (`scripts/download_datasets.py`) for UCI Adult Income, Bank Marketing, and Credit Default.
- [x] Implement schema inference & automatic column typing (`src/tabsynth/data/schema.py`).
- [x] Implement zero-leakage preprocessing pipeline (`src/tabsynth/data/preprocessor.py`):
  - Stratified 70/15/15 train/val/test splits.
  - Imputation & `QuantileTransformer` (normalizing to $\mathcal{N}(0, 1)$) fitted strictly on $D_{\text{train}}$.
  - Categorical ordinal & one-hot encoding with `<missing>` and `<unk>` tokens.
  - Invertible post-processing (`inverse_transform`).
- [x] Implement PyTorch `TabularDataset` and memory-aware `DataLoader` with MPS support (`src/tabsynth/data/dataset.py`).

### Phase 2: Dual Diffusion Engine (Gaussian & Multinomial D3PM)
- [x] Implement noise variance schedules (`src/tabsynth/diffusion/schedules.py`):
  - Cosine variance schedule (Nichol & Dhariwal, 2021).
  - Linear variance schedule.
- [x] Implement Continuous Gaussian Diffusion module (`src/tabsynth/diffusion/gaussian.py`):
  - Forward $q(\mathbf{x}_t | \mathbf{x}_0)$ noise injection with cumulative alphas $\bar{\alpha}_t$.
  - Tractable Gaussian reverse posterior $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$.
  - Predicted noise $\hat{\boldsymbol{\epsilon}}_\theta \to \hat{\mathbf{x}}_0^{\text{num}}$ reconstruction.
  - Single-step and iterative reverse denoising sampler.
- [x] Implement Discrete Multinomial Diffusion module (`src/tabsynth/diffusion/multinomial.py`):
  - Forward transition matrices $\mathbf{Q}_t$ and cumulative products $\bar{\mathbf{Q}}_t$.
  - Discrete forward corruption kernel $q(\mathbf{x}_t^{\text{cat}} | \mathbf{x}_0^{\text{cat}})$.
  - Exact reverse posterior calculation $q(\mathbf{x}_{t-1}^{\text{cat}} | \mathbf{x}_t^{\text{cat}}, \mathbf{x}_0^{\text{cat}})$.
  - Reverse categorical posterior sampling over probability simplices.
- [x] Unit tests for diffusion kernels (`tests/unit/test_diffusion_math.py`):
  - Test row-stochasticity of transition matrices ($\sum_j Q_{i,j} = 1.0$).
  - Test posterior probability normalization ($\sum_k \tilde{p}_k = 1.0$).
  - Test Gaussian schedule bounds ($0 < \beta_t < 1$).

### Phase 3: Hybrid Transformer-MLP Denoising Architecture
- [x] Implement Feature Tokenizer (`src/tabsynth/models/embeddings.py`):
  - Numerical scalar linear projections to $d_{\text{model}}$.
  - Categorical embedding tables to $d_{\text{model}}$.
  - Column identity embeddings $\mathbf{E}_{\text{col}} \in \mathbb{R}^{N_{\text{cols}} \times d_{\text{model}}}$.
  - Sinusoidal timestep embeddings and target-class conditioning embeddings.
- [x] Implement Transformer Feature Interaction Encoder (`src/tabsynth/models/transformer.py`):
  - Pre-LN Transformer encoder blocks with Multi-Head Self-Attention.
  - Feed-forward networks with GELU and dropout.
- [x] Implement Residual MLP Denoiser (`src/tabsynth/models/denoiser.py`):
  - Residual dense blocks with LayerNorm and SiLU.
  - Continuous noise prediction head $\hat{\boldsymbol{\epsilon}}_\theta \in \mathbb{R}^{N_{\text{num}}}$.
  - Independent categorical clean-state logit heads $\hat{\mathbf{p}}_{0, j}^{\text{cat}} \in \mathbb{R}^{K_j}$.
- [x] Implement Full Hybrid Model assembly (`src/tabsynth/models/hybrid.py`).
- [x] Implement MLP-Only Diffusion Baseline (`src/tabsynth/baselines/mlp_diffusion.py`) matching TabDDPM.
- [x] Unit tests for tensor shapes, parameter counts, and forward passes (`tests/unit/test_models.py`).

### Phase 4: Training Pipeline, Combined Losses & Conditional Generation
- [x] Implement Composite Diffusion Loss (`src/tabsynth/training/losses.py`):
  - Numerical Gaussian MSE loss $\mathcal{L}_{\text{num}}$.
  - Categorical D3PM KL divergence loss $\mathcal{L}_{\text{cat}}$.
  - Auxiliary clean categorical cross-entropy $\mathcal{L}_{\text{aux}}$.
  - $\mathcal{L}_{\text{total}} = \lambda_{\text{num}} \mathcal{L}_{\text{num}} + \lambda_{\text{cat}} \mathcal{L}_{\text{cat}} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}$.
- [x] Implement Training Manager (`src/tabsynth/training/trainer.py`):
  - Configuration-driven training loop with AdamW and cosine learning rate decay.
  - Dynamic device selection (MPS / CPU / CUDA).
  - Gradient clipping (1.0) and validation loss checkpoint saving.
  - Memory cleanup via `torch.mps.empty_cache()` every epoch.
- [x] Implement Reverse Sampling Engine (`src/tabsynth/sampling/sampler.py`):
  - Joint continuous-discrete reverse sampling.
  - Classifier-Free Guidance (CFG) extrapolation.
  - Balanced condition sampling for minority-class synthesis.
- [x] Execute smoke test and initial model training on Adult Income dataset.
- [x] Verify generated synthetic CSV format and validity.

### Phase 5: Comprehensive 3D Evaluation Suite & Research Experiments
- [x] Implement Statistical Fidelity Module (`src/tabsynth/evaluation/fidelity.py`):
  - Two-sample Kolmogorov-Smirnov (KS) test for continuous variables.
  - Total Variation Distance (TVD) for discrete variables.
  - Wasserstein-1 distance.
  - Correlation Matrix Frobenius distance ($D_{\text{corr}}$).
  - Classifier Two-Sample Test (C2ST).
- [x] Implement Downstream ML Utility Module (`src/tabsynth/evaluation/utility.py`):
  - TSTR (Train on Synthetic, Test on Real) vs TRTR (Train on Real, Test on Real).
  - Classifiers: Logistic Regression, Random Forest, XGBoost.
  - Metrics: Macro-F1, Balanced Accuracy, AUROC, AUPRC, Minority Recall.
- [x] Implement Empirical Privacy Auditing Module (`src/tabsynth/evaluation/privacy.py`):
  - Distance-to-Closest-Record (DCR) distribution ($R_{\text{train}}$ vs $R_{\text{test}}$).
  - Nearest Neighbor Distance Ratio (NNDR).
  - Distance-based and Shadow-Model Membership Inference Attacks (MIA).
  - Attribute Inference evaluation.
- [x] Implement Baseline Comparison Harness (`src/tabsynth/baselines/`):
  - Gaussian Copula baseline (`src/tabsynth/baselines/copula.py`).
  - TabDDPM MLP-only baseline (`src/tabsynth/baselines/mlp_diffusion.py`).
- [x] Implement Experiment Runner (`experiments/run_experiment.py`):
  - Automated execution of Experiments E1 through E9 with random seeds.

### Phase 6: REST API, Streamlit Dashboard & Deployment
- [x] Build FastAPI Backend (`api/server.py`, `api/routes.py`, `api/schemas.py`):
  - Ingestion, training status, generation, and evaluation endpoints.
- [x] Build Streamlit "TabSynth Analytics" UI (`app/main.py`):
  - Multi-tab research dashboard with live metrics, Plotly charts, and export.
- [x] Configure Docker & Containerization (`Dockerfile`, `docker-compose.yml`).
- [x] End-to-end integration and API tests passing (18/18 tests passing).
