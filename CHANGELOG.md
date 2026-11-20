# Changelog

All notable changes to the **TabSynth AI** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.9.0-rc.1] - 2026-11-18

### Added
- **Phase 08: Final Release Audit, Scientific Validation, Reproducibility & Portfolio Launch**
  - **Comprehensive Project Audit** (`docs/phase-08/FINAL_PROJECT_AUDIT.md`): End-to-end review of ML core, API server, persistence layer, Docker configuration, and test suites.
  - **Scientific Validation Report** (`docs/phase-08/SCIENTIFIC_VALIDATION_REPORT.md`): Rigorous mathematical derivation of Gaussian and discrete D3PM diffusion forward/reverse processes, Bayes posterior equations, Classifier-Free Guidance (CFG) score interpolation, Pre-LN cross-feature self-attention, and composite loss formulations.
  - **Research Reproducibility Audit** (`docs/phase-08/REPRODUCIBILITY_REPORT.md`): Verified deterministic 70/15/15 stratified dataset split protocol, seed control (`seed=42`), Pydantic configuration hierarchy, checkpoint serialization/restoration, and step-by-step reproduction runbook.
  - **Consolidated Empirical Benchmark Report & Data Exports** (`docs/phase-08/FINAL_BENCHMARK_REPORT.md`):
    - Exported 7 structured datasets to `experiments/results/`: `benchmark_summary.csv`, `benchmark_summary.json`, `fidelity_summary.csv`, `utility_summary.csv`, `privacy_summary.csv`, `ablation_summary.csv`, `performance_summary.csv`.
    - Multi-dimensional comparison across Gaussian Copula, TVAE, TabDDPM MLP, and TabSynth Hybrid on the Adult Income dataset ($N=32,561$).
    - Pearson correlation Frobenius error: TabSynth Hybrid (0.1117) vs TVAE (0.6326) and TabDDPM MLP (0.1591).
    - Class-conditional sampling ablation (E5): +48.55% minority recall gain from balanced conditioning.
    - Privacy protection: DCR ratio 1.0155, 0 exact matches, MIA ROC-AUC 0.4907.
  - **Application Acceptance Test Suite** (`scripts/run_phase08_acceptance.py` & `docs/phase-08/APPLICATION_ACCEPTANCE_REPORT.md`):
    - Programmatic verification of 23 primary user workflows (W01–W23) and 11 boundary/failure cases (B01–B11) via FastAPI TestClient and Streamlit AppTest.
    - 100% acceptance pass rate (34 / 34 tests passing).
  - **Pydantic Validation Hardening**: Enforced `gt=0` constraints on hyperparameters (`epochs`, `diffusion_steps`, `learning_rate`) and sampling requests (`num_samples`).
  - **Architecture Diagrams**: 8 detailed Mermaid diagrams in `docs/phase-08/figures/`.
  - **Demonstration Guide & Technical Presentation** (`docs/phase-08/DEMO_GUIDE.md`, `docs/phase-08/PROJECT_PRESENTATION.md`).
  - **Interview Preparation Guide** (`docs/phase-08/INTERVIEW_PREPARATION.md`): In-depth technical Q&A covering diffusion math, Pre-LN attention, CFG, evaluation metrics, and privacy auditing.
  - **Release Security Audit** (`docs/phase-08/RELEASE_SECURITY_AUDIT.md`): Secret scanning, dependency auditing, license compliance, and path confinement verification.
  - **Standardized Metadata**: Added `CITATION.cff` for academic attribution.
  - Expanded test suite to **128 unit and integration tests**, all passing deterministically.

---

## [0.7.0] - 2026-11-17

### Added
- **Phase 07: Production Hardening, Security & Deployment Engineering**
  - **Confinement & Path Traversal Guards** (`api/utils/security.py`): Strict containment verifying canonical paths within registered roots (`data/`, `checkpoints/`, `artifacts/`).
  - **Upload Sanitization & Extension Guards**: File size limiting (`MAX_UPLOAD_SIZE_BYTES = 100MB`), null byte stripping, binary content sniffing, and safe extension enforcement.
  - **SQLite WAL & High-Concurrency Pragmas** (`api/db/session.py`): Configured `journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=10000`, `foreign_keys=ON`, and `cache_size=-64000`.
  - **Hot Online Database Backup & Restore** (`scripts/backup_db.py`, `scripts/restore_db.py`): Uses `sqlite3.backup()` API for hot live snapshots with microsecond-precision timestamps and integrity verification.
  - **Crash Recovery & Orphan Job Sweep** (`recover_orphaned_jobs`): Automatically recovers hanging or orphaned `running`/`queued` background jobs upon server restart.
  - **Multi-Stage Production Dockerfile & Compose** (`Dockerfile`, `docker-compose.yml`): Non-root execution (`uid=10001`), shared persistence volumes, and automated healthchecks.
  - **Phase 07 Production Smoke Suite** (`scripts/run_phase07_smoke.py`): End-to-end verification of security, backup, recovery, and configuration.

---

## [0.6.0] - 2026-11-16

### Added
- **Phase 06: Full-Stack Application & Interactive Platform**
  - **FastAPI REST API Server** (`api/`): High-performance asynchronous backend providing endpoints for dataset management, model training, reverse generation, evaluation, and background job polling.
  - **Thread-Safe Background Job Manager** (`api/services/job_service.py`): Asynchronous worker management with live progress tracking, step telemetry, and cooperative cancellation tokens.
  - **10-View Streamlit Research SaaS Dashboard** (`app/main.py`):
    1. Overview Dashboard: High-level KPI cards, telemetry, and architecture diagram.
    2. Dataset Manager: Dataset upload, profiling, and schema exploration.
    3. Data Explorer: Marginal distributions, correlation heatmaps, and outlier inspection.
    4. Model Studio: Diffusion configuration, hyperparameter tuning, and training dispatch.
    5. Synthetic Data Generator: Sampling trajectory controls, CFG guidance scale, and CSV export.
    6. Fidelity Analysis: Marginal overlays, correlation differences, and C2ST distinguishability.
    7. Utility Evaluation: TSTR vs TRTR downstream classifier benchmarks and parity ratios.
    8. Privacy Audit: Gower DCR distributions, NNDR histograms, and shadow MIA ROC curves.
    9. Benchmark Center: Multi-model comparison matrix, radar charts, and ranking leaderboards.
    10. Experiment Explorer: Historical run comparisons and artifact inspection.
  - **Enterprise Dark Technical Theme** (`app/styles.py`): Custom CSS variables, responsive containers, status badges, and Vega-Lite / ECharts visual styling.
  - **API Client Layer** (`app/api_client.py`): Resilient client communicating between Streamlit and FastAPI.
  - **Phase 06 End-to-End Application Smoke Suite** (`scripts/run_phase06_smoke.py`).

---

## [0.1.0] - 2026-11-15

### Added
- **Phase 01: Research Foundation & Data Pipeline**
  - Dataset ingestion and preparation scripts for UCI Adult Income, Bank Marketing, and Credit Default.
  - Automated schema inference (`TableSchema`, `ColumnMetadata`) with type classification.
  - Leak-free tabular preprocessing (`TabularPreprocessor`) with bidirectional numerical scaling and categorical encoding.
  - Stratified 70/15/15 partitioning protocol.
- **Phase 02: Mixed-Type Diffusion Engine**
  - Continuous Gaussian DDPM diffusion with Linear, Cosine, and Sigmoid noise schedules.
  - Discrete D3PM multinomial diffusion over categorical simplices with exact analytical Bayes posterior marginalization.
  - Unified `MixedTypeDiffusion` orchestrating continuous and discrete transitions.
- **Phase 03: Hybrid Transformer-MLP Denoising Architecture**
  - Feature-wise numerical and categorical tokenizers with learnable column identity embeddings.
  - Pre-LayerNorm Transformer multi-head self-attention feature encoder.
  - Sinusoidal time embedding and Classifier-Free Guidance (CFG) class conditioning ($p_{\text{uncond}} = 0.15$).
  - Residual MLP denoising backbone with SwiGLU / SiLU non-linearities and dual output heads.
  - TabDDPM-style flat MLP denoiser baseline (`TabSynthMLPDenoiser`) for architectural ablations.
- **Phase 04: Training Pipeline & Sampling with CFG**
  - Parameter grouping optimizer separating weight decay for 2D+ matrices from 1D biases/norms.
  - Warmup + Cosine Annealing learning rate scheduler.
  - Early stopping with best weight restoration and rolling top-$k$ checkpointing (`keep_top_k: 2`).
  - Mixed ancestral reverse sampler (`MixedAncestralSampler`) with Classifier-Free Guidance extrapolation.
  - Class-conditional sampling strategies: `natural`, `balanced`, and `class_specific`.
  - Full Apple Silicon M4 MPS acceleration with CPU fallback.
- **Phase 05: Comprehensive Evaluation, Baselines & Privacy Auditing**
  - Statistical fidelity suite: Kolmogorov-Smirnov test, 1-Wasserstein distance, Total Variation Distance (TVD), Pearson and Spearman correlation Frobenius distance, Theil's U, and 5-fold cross-validated Classifier Two-Sample Test (C2ST).
  - Downstream machine learning utility suite: TRTR, TSTR, relative macro-F1 parity ratios, minority recall gain, and mixed-data minority augmentation across $[0.25\times, 0.5\times, 1.0\times, 2.0\times]$.
  - Empirical privacy auditing suite: Pairwise normalized Gower distance, Distance to Closest Record (DCR) to Train and Test partitions, DCR ratios, 5th percentile ratios, Nearest Neighbor Distance Ratios (NNDR), exact match audits, distance-based Membership Inference Attacks (MIA), Attribute Inference Attacks (AIA), and predicate-based singling out.
  - Standardized baseline registry (`BaseSynthesizer`) implementing Gaussian Copula, native PyTorch TVAE, TabDDPM MLP, and TabSynth Hybrid.
  - Interactive Plotly visualization suite generating standalone HTML and JSON figures.
  - Official research evaluation report (`docs/PHASE_05_EVALUATION_REPORT.md`) derived from empirical runs.
