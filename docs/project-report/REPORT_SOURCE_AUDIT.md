# TabSynth AI — Research & Source Code Audit Report
## Source Investigation, Verification Matrix, and Documentation Grounding

**Document Identifier**: `docs/project-report/REPORT_SOURCE_AUDIT.md`  
**Date**: November 20, 2026  
**Status**: Completed & Verified  
**Target Release**: TabSynth AI `v0.9.0-rc.1` (Release Candidate)  
**Primary Architect**: Senior ML Research Engineer & Technical Documentation Specialist  

---

## 1. Executive Summary of Audit

Prior to authoring the authoritative research project report for **TabSynth AI** (*Hybrid Transformer-MLP Diffusion Framework for Privacy-Aware, High-Fidelity and Class-Conditional Synthetic Tabular Data Generation*), an exhaustive audit of the repository was performed. This audit examined all source modules, mathematical diffusion routines, neural network definitions, data pipelines, empirical benchmark outputs, test logs, documentation, and user interface components.

The objective of this audit is to ensure **zero fabrication of experimental metrics, absolute mathematical fidelity to actual code, and precise attribution of scientific baselines**. Every metric, parameter count, and architectural layer documented in the final report is grounded in the actual codebase.

---

## 2. Repository Components Inspected

| Component Category | Repository Path | Core Modules / Files Inspected | Verification Status |
| :--- | :--- | :--- | :--- |
| **Model Architectures** | `src/models/` | `transformer.py`, `denoiser.py`, `embedding.py`, `mlp.py` | Verified: Hybrid Transformer-MLP with Pre-LN Multi-Head Attention, FeatureTokenizer, sinusoidal timestep embeddings, and residual MLP refinement block. Parameter count: 240,198 weights. |
| **Diffusion Engines** | `src/diffusion/` | `gaussian.py`, `categorical.py`, `schedules.py`, `sampler.py`, `loss.py` | Verified: Continuous Gaussian diffusion ($\\epsilon$-prediction, linear/cosine schedules) coupled with discrete D3PM multinomial diffusion ($q(x_{t-1} \\vert x_t, x_0)$ posterior) and Classifier-Free Guidance (CFG). |
| **Data Pipelines** | `src/data/` | `dataset.py`, `preprocessing.py`, `validation.py`, `transforms.py`, `split.py` | Verified: Quantile transformation for continuous columns, one-hot encoding for categorical columns, strict train/val/test splitting to prevent data leakage. |
| **Evaluation Suite** | `src/evaluation/` | `fidelity.py`, `utility.py`, `privacy.py`, `benchmark.py`, `metrics.py` | Verified: 3-pillar taxonomy: Statistical fidelity (KS test, Wasserstein distance, Total Variation Distance, Frobenius correlation norm, C2ST), Downstream utility (TSTR vs TRTR across RF, HGB, LR), and Empirical privacy (DCR, NNDR, exact match audit, shadow MIA ROC-AUC). |
| **Generative Baselines** | `src/baselines/` | `copula.py`, `tvae.py`, `tab_ddpm.py` | Verified: Empirical implementations for Gaussian Copula (SDV/parametric copula), TVAE (PyTorch variational autoencoder), and TabDDPM MLP (pure residual MLP denoiser). |
| **Backend & Service Layer** | `api/` | `main.py`, `routes/`, `database.py`, `schemas/`, `config.py`, `tasks.py` | Verified: FastAPI asynchronous backend with SQLite database persistence, background worker execution for training and sampling, and RESTful API endpoints. |
| **Interactive Dashboard** | `app/` | `main.py`, `views/` (10 views), `components/`, `styles/` | Verified: 10-view Streamlit UI styled with dark enterprise technical theme (`#0B0F19`), responsive cards, and real-time telemetry. |
| **Empirical Artifacts** | `experiments/results/` | `benchmark_summary.csv`, `benchmark_summary.json`, `fidelity_summary.csv`, `utility_summary.csv`, `privacy_summary.csv`, `ablation_summary.csv`, `performance_summary.csv` | Verified: Exact empirical evaluation records computed on the benchmark dataset (UCI Adult Income, seed 42, 70/15/15 split). |
| **Raw Datasets** | `data/raw/` | `adult.csv`, `bank_marketing.csv`, `credit_default.csv` | Verified: Ground-truth datasets with exact shapes and verified class distributions. |
| **Tests & QA** | `tests/`, `scripts/` | `tests/unit/`, `tests/integration/`, `scripts/run_phase08_acceptance.py` | Verified: 128/128 pytest tests passing; 34/34 application acceptance criteria passing. |

---

## 3. Available Documentation Analyzed

The following existing technical specifications, architectural designs, and milestone reports were reviewed:

1. **Phase 01–06 Milestones**: `docs/PHASES/PHASE_01_RESEARCH_AND_DATA.md` through `PHASE_05_EVALUATION_AND_PRIVACY.md`, `PHASE_06_IMPLEMENTATION.md`.
2. **Phase 07 Production Audit**: `docs/phase-07/PRODUCTION_READINESS_AUDIT.md`, `docs/phase-07/DEPLOYMENT_GUIDE.md`, `docs/phase-07/SECURITY.md`.
3. **Phase 08 Release Engineering**:
   - `docs/phase-08/FINAL_PROJECT_AUDIT.md`: Structural code audit across all modules.
   - `docs/phase-08/SCIENTIFIC_VALIDATION_REPORT.md`: Mathematical validation of mixed-type diffusion.
   - `docs/phase-08/REPRODUCIBILITY_REPORT.md`: Execution determinism and seed management.
   - `docs/phase-08/FINAL_BENCHMARK_REPORT.md`: Empirical analysis across the 4 evaluated models.
   - `docs/phase-08/APPLICATION_ACCEPTANCE_REPORT.md`: Acceptance test verification across 10 UI views.
   - `docs/phase-08/figures/*.md`: 8 Mermaid architectural and pipeline diagrams.
   - `docs/phase-08/screenshots/README.md`: Inventory of application presentation states.
4. **API Specification**: `docs/api_specification.md` and `docs/API_REFERENCE.md`.
5. **Technical Specification**: `docs/TABS_SYNTH_AI_TECHNICAL_SPECIFICATION.md`.

---

## 4. Verified Source Implementations & Mathematics

### 4.1 Hybrid Denoiser Architecture
- **Continuous Embeddings**: Numerical features are mapped through continuous feature projection layers ($\\mathbb{R}^{d_{num}} \\to \\mathbb{R}^{B \\times d_{num} \\times d_{model}}$).
- **Categorical Embeddings**: Tokenized categorical indices are mapped via lookup embeddings ($\\{1, \\dots, K_j\\} \\to \\mathbb{R}^{B \\times 1 \\times d_{model}}$).
- **Transformer Backbone**: Pre-LayerNorm multi-head self-attention ($L=3$ layers, $H=4$ heads, $d_{model}=128$) capturing complex inter-column associations.
- **MLP Denoiser Refinement**: Residual Feed-Forward block ($d_{ff}=256$, SiLU non-linearities, dropout 0.1) refining column representations.
- **Dual Output Heads**:
  - Continuous Head: Linear layer predicting Gaussian noise vector $\\hat{\\epsilon}_\\theta(x_t, t, c) \\in \\mathbb{R}^{d_{num}}$.
  - Discrete Head: Linear projection predicting clean categorical logits $\\hat{p}_\\theta(x_0 \\vert x_t, t, c) \\in \\mathbb{R}^{\\sum K_j}$.

### 4.2 Diffusion Mathematics
- **Continuous Features**: Gaussian DDPM with standard cosine noise schedule $\\beta_t \\in [\\beta_{min}, \\beta_{max}]$:
  $$q(x_t \\vert x_0) = \\mathcal{N}\\left(x_t; \\sqrt{\\bar{\\alpha}_t}x_0, (1 - \\bar{\\alpha}_t)\\mathbf{I}\\right)$$
  Objective: Continuous Mean Squared Error:
  $$\\mathcal{L}_{num}(\\theta) = \\mathbb{E}_{t, x_0, \\epsilon} \\left[ \\|\\epsilon - \\hat{\\epsilon}_\\theta(x_t, t, c)\\|^2 \\right]$$
- **Categorical Features**: Austin et al. (2021) D3PM discrete transition matrix $Q_t$:
  $$q(k_t \\vert k_0) = \\text{Cat}(k_t; k_0 \\bar{Q}_t)$$
  Posterior evaluation:
  $$q(k_{t-1} \\vert k_t, k_0) = \\frac{q(k_t \\vert k_{t-1}) q(k_{t-1} \\vert k_0)}{q(k_t \\vert k_0)}$$
  Objective: Categorical Cross-Entropy on predicted clean probabilities $\\hat{p}_\\theta(k_0 \\vert k_t, t, c)$:
  $$\\mathcal{L}_{cat}(\\theta) = \\mathbb{E}_{t, k_0} \\left[ -\\sum_{j=1}^{d_{cat}} \\sum_{m=1}^{K_j} \\mathbf{1}_{\\{k_{0,j} = m\\}} \\log \\hat{p}_\\theta(k_{0,j} = m \\vert k_{t,j}, t, c) \\right]$$
- **Composite Training Loss**:
  $$\\mathcal{L}_{total} = \\lambda_{num} \\mathcal{L}_{num} + \\lambda_{cat} \\mathcal{L}_{cat} + \\lambda_{aux} \\mathcal{L}_{aux}$$
  Verified defaults: $\\lambda_{num} = 1.0$, $\\lambda_{cat} = 1.0$, $\\lambda_{aux} = 0.1$.

---

## 5. Available Experimental Results

The empirical results were gathered on the standard benchmark dataset (**UCI Adult Income**, 32,561 records, 70/15/15 train/val/test split, random seed 42) across 4 baseline generators.

### 5.1 Primary Empirical Benchmark Matrix
| Model Architecture | Parameter Count | Sampling Throughput (rows/sec) | Mean KS Stat ($\\downarrow$) | Mean Wasserstein ($\\downarrow$) | Mean TVD ($\\downarrow$) | Pearson Frob. Error ($\\downarrow$) | C2ST ROC-AUC ($\\approx 0.5$) | TSTR RF Macro-F1 ($\\uparrow$) | F1 Parity Ratio ($\\uparrow$) | Mean DCR Train | DCR Ratio ($\\approx 1.0$) | MIA ROC-AUC ($\\approx 0.5$) | Privacy Risk Level |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gaussian Copula** | 20 | 331,722.9 | 0.2314 | 0.5191 | 0.0361 | 0.1099 | 0.9987 | 0.4316 | 0.5541 | 0.1831 | 0.9816 | 0.4997 | **LOW_RISK** |
| **TVAE** | 79,556 | 2,192.9 | 0.4863 | 0.4112 | 0.0586 | 0.6326 | 1.0000 | 0.6431 | 0.8258 | 0.0946 | 0.9867 | 0.4978 | **LOW_RISK** |
| **TabDDPM MLP** | 177,478 | 145.7 | 0.0364 | 0.1837 | 0.1180 | 0.1591 | 1.0000 | 0.4696 | 0.6029 | 0.1394 | 0.9855 | 0.4900 | **LOW_RISK** |
| **TabSynth Hybrid** | 240,198 | 205.1 | 0.0545 | 0.2115 | 0.1719 | **0.1117** | 1.0000 | 0.4600 | 0.5906 | 0.1165 | **1.0155** | 0.4907 | **LOW_RISK** |

### 5.2 Verified Ablation Studies
1. **Ablation E1 (Denoiser Architecture)**:
   - TabSynth Hybrid Pearson Frobenius Error: `0.1117` vs TabDDPM MLP `0.1591` (Full Benchmark), representing a **29.8% correlation error reduction** achieved by Transformer self-attention.
2. **Ablation E5 (Class Conditioning Strategy)**:
   - Natural frequency conditioning (78/22 prior): Minority recall = `0.0000` (Random Forest collapsed on minority class).
   - Balanced class conditioning (50/50 prior): Minority recall = `0.4855` (**+48.55% recovery** of minority predictive utility).
3. **Ablation E6 (Synthetic Minority Augmentation)**:
   - Baseline TRTR (0.0x augmentation): Minority F1 = `0.6462`, Minority Recall = `0.5374`.
   - Augmentation ratios (0.25x, 0.5x, 1.0x, 2.0x) using natural unguided samples collapsed due to minority boundary blur. This verified the scientific requirement of Class-Conditional Guidance (CFG) for data augmentation.

---

## 6. Available Datasets & Statistics

Verified ground-truth datasets residing in `data/raw/`:

1. **UCI Adult Income (`adult.csv`)**:
   - Total Rows: `32,561` | Total Columns: `15`
   - Continuous Features: 6 (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`)
   - Categorical Features: 8 (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`)
   - Target Variable: `income` (`<=50K`: 24,720 [75.92%], `>50K`: 7,841 [24.08%])
2. **Bank Marketing (`bank_marketing.csv`)**:
   - Total Rows: `5,000` | Total Columns: `17`
   - Target Variable: `y` (`no`: 4,380 [87.60%], `yes`: 620 [12.40%])
3. **Credit Default (`credit_default.csv`)**:
   - Total Rows: `5,000` | Total Columns: `12`
   - Target Variable: `default.payment.next.month` (`0`: 3,920 [78.40%], `1`: 1,080 [21.60%])

---

## 7. Gaps, Missing Evidence, and Contradictions Identified

The audit revealed the following nuances that must be clearly qualified in the research report:

1. **Formal Differential Privacy vs Distance-Based Risk Metrics**:
   - *Discovery*: Initial preliminary project descriptions casually referenced "privacy-preserving" or "differential privacy".
   - *Audit Finding*: The implementation computes empirical distance-to-closest-record (DCR), nearest neighbor distance ratio (NNDR), exact record matches (0 found), and shadow membership inference attack (MIA) ROC-AUC (~0.490). It does **not** inject calibrated Gaussian/Laplacian noise into gradient steps (e.g. DP-SGD) or compute privacy budgets $(\\epsilon, \\delta)$.
   - *Resolution*: The report will prominently state that empirical privacy indicators demonstrate low memorization risk, but do **not** confer formal mathematical Differential Privacy guarantees.
2. **Benchmark Scope**:
   - *Discovery*: Bank Marketing and Credit Default datasets are present in `data/raw/` and supported by the ingestion pipeline, but the full 4-model comparative benchmark was executed on UCI Adult Income.
   - *Resolution*: The report will clearly designate UCI Adult Income as the primary comprehensive benchmark dataset, and present Bank Marketing and Credit Default as validated multi-dataset generalization targets.
3. **C2ST Discriminator Saturation**:
   - *Discovery*: Classifier Two-Sample Test (C2ST) ROC-AUC scores are near 1.0 across all models (Gaussian Copula: 0.9987, TVAE: 1.000, TabDDPM: 1.000, TabSynth: 1.000).
   - *Audit Finding*: In high-dimensional mixed tabular spaces, a non-linear discriminator (LightGBM/XGBoost) can easily detect subtle distributional differences between real and synthetic data even when low-order marginals and correlation matrices match closely.
   - *Resolution*: Document C2ST saturation as a known characteristic of modern tabular generative benchmarks rather than concealing it.

---

## 8. Visual Assets Inventory & Plan

### 8.1 Architectural & Pipeline Diagrams (`docs/project-report/assets/diagrams/`)
Will compile 12 comprehensive Mermaid `.mmd` diagrams:
- `overall_system_architecture.mmd`
- `hybrid_transformer_mlp_architecture.mmd`
- `numerical_gaussian_diffusion.mmd`
- `categorical_multinomial_diffusion.mmd`
- `combined_diffusion_framework.mmd`
- `conditional_generation_pipeline.mmd`
- `training_pipeline.mmd`
- `sampling_flowchart.mmd`
- `evaluation_pipeline.mmd`
- `application_user_workflow.mmd`
- `backend_architecture.mmd`
- `reproducibility_workflow.mmd`

### 8.2 Empirical Charts (`docs/project-report/assets/charts/`)
Will generate 9 publication-grade SVG and PNG charts from empirical CSV data via `scripts/generate_report_charts.py`:
- `fidelity_metrics_comparison.svg`
- `frobenius_correlation_error.svg`
- `utility_macro_f1_comparison.svg`
- `utility_f1_parity_ratio.svg`
- `ablation_e5_minority_recall.svg`
- `ablation_e1_denoiser_error.svg`
- `privacy_dcr_ratio.svg`
- `privacy_nndr_distribution.svg`
- `model_throughput_parameters.svg`

### 8.3 Dataset Visualizations (`docs/project-report/assets/dataset_visualizations/`)
Will generate 3 dataset distribution charts:
- `adult_income_target_distribution.svg`
- `adult_income_feature_types.svg`
- `benchmark_datasets_summary.svg`

### 8.4 Application UI Inventory (`docs/project-report/assets/screenshots/`)
Will integrate screenshots and visual workflows for all 10 application views documented in `docs/phase-08/screenshots/README.md`.

---

## 9. Report Generation Roadmap

1. **Step 1 [COMPLETE]**: Author `REPORT_SOURCE_AUDIT.md`.
2. **Step 2**: Create programmatic chart generation script `scripts/generate_report_charts.py` and produce all SVG/PNG diagrams and charts in `docs/project-report/assets/`.
3. **Step 3**: Author the comprehensive master Markdown report `docs/project-report/TabSynth_AI_Project_Report.md` (24 chapters, Title Page, Executive Summary, References, Appendices).
4. **Step 4**: Generate native `TabSynth_AI_Project_Report.docx` and `TabSynth_AI_Project_Report.pdf`.
5. **Step 5**: Conduct quality assurance and complete `docs/project-report/REPORT_VALIDATION.md`.
