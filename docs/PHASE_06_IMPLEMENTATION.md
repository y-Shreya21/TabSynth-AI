# Phase 06 Implementation: Production TabSynth AI Application

## 1. Overview

In Phase 06, **TabSynth AI** (*Hybrid Transformer-MLP Diffusion Framework for Privacy-Aware, High-Fidelity and Class-Conditional Synthetic Tabular Data Generation*) was transitioned from a local research codebase into a production-grade multi-service software application.

The application architecture introduces:
1. **SQLite Persistent Metadata Storage**: Thread-safe relational persistence tracking datasets, trained models, asynchronous jobs, artifacts, and benchmark experiments.
2. **Asynchronous Background Job Engine**: Thread-pool execution engine with state transitions (`QUEUED` -> `RUNNING` -> `COMPLETED` / `FAILED` / `CANCELLED`), progress tracking, and hardware concurrency bounds.
3. **Domain Service Layer**: Decoupled domain services (`DatasetService`, `ModelService`, `GenerationService`, `EvaluationService`, `ExperimentService`, `JobManager`) bridging HTTP routes with underlying PyTorch diffusion algorithms.
4. **Modular FastAPI REST API**: Endpoints for dataset ingestion, schema profiling, model training, reverse diffusion generation, fidelity/utility/privacy benchmarks, and job management.
5. **Polished 8-Page Streamlit SaaS Interface ("TabSynth Analytics")**: Interactive web dashboard providing visual exploratory analysis, training orchestration, conditional sampling, and scientific evaluation dashboards.
6. **Containerization & Deployment**: Docker multi-service composition (`docker-compose.yml`), non-root application execution, and configurable environment controls.

---

## 2. Architecture & Design Patterns

### 2.1 Persistence Layer (`api/db/`)
- Implemented with Python's standard `sqlite3` using Write-Ahead Logging (`PRAGMA journal_mode=WAL`) and foreign keys (`PRAGMA foreign_keys=ON`).
- **Entity Schemas:**
  - `datasets`: Ingested dataset metadata, row counts, continuous/categorical columns, cardinalities, missing-value maps, and class frequencies.
  - `models`: Trained checkpoints, hyperparameter configurations, validation losses, and target variables.
  - `jobs`: Background task tracking, lifecycle status, floating-point progress (`0.0 - 1.0`), step counts, execution errors, and result payloads.
  - `experiments`: Scientific benchmark logs, fidelity KS scores, TSTR utility parity, and empirical privacy metrics.
  - `artifacts`: Tracked references to generated synthetic CSV files, PyTorch checkpoints, and evaluation reports.

### 2.2 Background Job Management (`api/services/job_service.py`)
- Standard FastAPI background tasks lack built-in status querying, persistence, and cancellation.
- The `JobManager` singleton introduces a bounded `ThreadPoolExecutor` with a `threading.Semaphore` limiting concurrent GPU/MPS operations to prevent out-of-memory errors on Apple Silicon and shared hardware.
- Safe task cancellation is achieved via `threading.Event` checked at epoch and sampling step boundaries.

### 2.3 Modular REST Routing (`api/routes/`)
- Converted monolithic prototypes into organized router modules:
  - `api/routes/health.py`: Liveness (`/health`), readiness (`/ready`), and version (`/version`).
  - `api/routes/datasets.py`: File upload, schema profiling, listing, and inspection.
  - `api/routes/models.py`: Asynchronous training launch and checkpoint status polling.
  - `api/routes/generation.py`: Reverse diffusion sampling with Classifier-Free Guidance.
  - `api/routes/evaluation.py`: Statistical fidelity, downstream utility, and privacy audits.
  - `api/routes/experiments.py`: Historical experiment log inspection.
  - `api/routes/jobs.py`: Task progress monitoring and cancellation.

### 2.4 TabSynth Analytics Dashboard (`app/main.py`)
- Structured across 8 pages:
  1. **Overview Dashboard**: High-level system statistics, hardware accelerator detection, recent execution history.
  2. **Dataset Manager**: Drag-and-drop CSV upload, schema inference, continuous/categorical separation, and missing-value profiling.
  3. **Model Training**: Hyperparameter configuration, architecture selection (Hybrid Transformer-MLP vs MLP baseline), device selection, and live training progress.
  4. **Synthetic Data Generator**: Checkpoint selection, class-conditioning strategies (balanced, natural, class-specific, unconditional), CFG scale adjustment, and CSV export.
  5. **Fidelity Dashboard**: Scorecard metrics (KS, TVD, Frobenius correlation distance, C2ST ROC-AUC), interactive 1D marginal density histograms, and categorical frequency bar charts.
  6. **Utility Dashboard**: Train-on-Synthetic, Test-on-Real (TSTR) versus Train-on-Real, Test-on-Real (TRTR) comparative utility and minority recall delta.
  7. **Privacy Audit**: Distance-to-Closest-Record (DCR) ratios, exact matches check, shadow Membership Inference Attack (MIA) AUROC, and differential privacy disclaimer.
  8. **Experiment Explorer**: Comparative table across historical runs, JSON benchmark viewer, and export tools.

---

## 3. Verification & Empirical Results

1. **Phase 06 End-to-End Smoke Test (`scripts/run_phase06_smoke.py`)**:
   - Passed in **3.81s** (< 30s threshold).
   - Validated database initialization, dataset ingestion, asynchronous training, checkpoint saving, synthetic sampling, fidelity scoring, and REST endpoints.
2. **Phase 05 Smoke Benchmark (`scripts/run_phase05_smoke.py`)**:
   - Passed in **1.66s**.
3. **Comprehensive Pytest Suite (`tests/`)**:
   - **120 / 120 tests passed** in **17.56s** (100% pass rate).
   - Zero regressions across Phases 01–05.
