# TabSynth Analytics: Production Readiness & Architecture Audit

**Project:** TabSynth AI / TabSynth Analytics  
**Research Title:** *Hybrid Transformer-MLP Diffusion Framework for Privacy-Aware, High-Fidelity and Class-Conditional Synthetic Tabular Data Generation*  
**Phase:** 07 — Production Hardening, Deployment & Release Engineering  
**Date:** November 2026  
**Auditor:** Senior DevOps & Production Reliability Engineering Team  

---

## 1. Executive Summary

This document presents a comprehensive technical audit of the **TabSynth Analytics** application following the completion of Phases 01 through 06. The platform integrates a mixed-type tabular diffusion generative model (Hybrid Transformer-MLP and baseline MLP architectures) with an asynchronous FastAPI REST service and a high-contrast Streamlit analytical dashboard.

The audit evaluates the platform's readiness for reliable containerized deployment, public technical demonstrations, and security compliance. While core machine learning and API services are functional and rigorously tested (120/120 passing unit/integration tests), several production hardening requirements, security safeguards, and deployment optimizations must be addressed before release.

---

## 2. Existing Architecture Overview

```
                      +------------------------------------------+
                      |         Web Client / Browser            |
                      +--------------------+---------------------+
                                           |
                    HTTP / WebSocket :8501 | HTTP REST :8000
                                           v
       +-----------------------------------+-----------------------------------+
       |                                                                       |
       v                                                                       v
+-------------------------------+                     +-------------------------------+
|     Streamlit Frontend        |                     |      FastAPI REST API         |
|   (app/main.py, styles.py,    |                     |   (api/server.py, routes/,    |
|   components/theme, cards)    |                     |    middleware/, schemas/)     |
+---------------+---------------+                     +---------------+---------------+
                |                                                     |
                | Inter-Process / Local DB                            | Async Execution
                v                                                     v
+-------------------------------+                     +-------------------------------+
|     SQLite Persistence        | <================== |     JobManager (Worker Pool)  |
|      (data/tabsynth.db)       |                     |  (ThreadPoolExecutor, bounded)|
|  - datasets, models, jobs,    |                     +---------------+---------------+
|    experiments, artifacts     |                                     |
+-------------------------------+                                     v
                |                                     +-------------------------------+
                v                                     |   TabSynth ML Engine (Core)   |
+-------------------------------+                     |  - Hybrid Transformer-MLP     |
|   File-System Storage Tree    |                     |  - Mixed Diffusion (Gauss/Cat)|
|  - data/raw/                  | <================== |  - Guidance & Conditioning    |
|  - data/synthetic/            |                     |  - Fidelity / Utility / Priv  |
|  - checkpoints/               |                     +-------------------------------+
|  - experiments/results/       |
+-------------------------------+
```

### Component Analysis:
1. **Core ML Framework (`src/tabsynth/`)**: Mixed-type diffusion engine combining Gaussian noise diffusion for continuous variables and multinomial transition diffusion for categorical variables. Class-conditional sampling via Classifier-Free Guidance (CFG). Supports Apple Silicon MPS acceleration with CPU fallback.
2. **REST API Service (`api/`)**: Built on FastAPI and Uvicorn. Implements asynchronous background task offloading (`JobManager`), request correlation ID tagging, structured logging, and schema validation via Pydantic v2 models.
3. **Analytical Frontend (`app/`)**: Built with Streamlit and Plotly. High-contrast dark theme matching target design mockups with real-time model training monitor, synthetic generator, and benchmark visualizers.
4. **Data & Metadata Persistence (`data/tabsynth.db`)**: SQLite 3 with Write-Ahead Logging (WAL) and foreign keys enabled. Manages records for datasets, trained models, job progress, and experimental evaluations.
5. **Artifact Storage**: Filesystem-based directory structure separating raw uploads (`data/raw/`), generated datasets (`data/synthetic/`), PyTorch checkpoints (`checkpoints/`), and serialized evaluation metrics (`experiments/results/`).

---

## 3. Deployment & Operational Limitations

| Area | Current Implementation | Production Risk / Limitation | Recommended Hardening |
| :--- | :--- | :--- | :--- |
| **Compute Hardware** | Local MPS / CUDA / CPU auto-selection | Linux Docker containers cannot access Apple Silicon MPS; will fail if MPS is forcefully configured. | Enforce CPU default inside containers; document hardware constraints clearly. |
| **Container Image** | Single-stage Dockerfile | Build dependencies (`gcc`, build tools) remain in final image; image size is larger than necessary. | Multi-stage Docker build separating build dependencies from minimal runtime environment. |
| **Docker Compose** | Bind-mounted host directories | Data files and SQLite database directly share host filesystem without named volume abstraction. | Provide production compose configuration with named persistent volumes and health check dependencies. |
| **Process Crash Recovery** | In-memory job state tracking | If API server crashes while a job is running, the job remains stuck in `running` status forever in SQLite. | Implement startup recovery sweep marking orphan `running`/`queued` jobs as `failed`. |
| **Database Backups** | Manual file copying of `tabsynth.db` | Copying an active SQLite database in WAL mode can result in corrupted backups if transactions are in-flight. | Add automated zero-downtime online backup utility using `sqlite3.Connection.backup` or `VACUUM INTO`. |

---

## 4. Security Audit & Vulnerability Assessment

### 4.1 Path Traversal (High Risk)
- **Finding**: Several API endpoints (e.g., `/datasets/schema`, `/evaluate/fidelity`, `/evaluate/utility`, `/evaluate/privacy`) accepted raw filesystem path parameters (`data_path`, `real_data_path`, `synthetic_data_path`) without verifying that the resolved path is contained within authorized application directories.
- **Impact**: Potential arbitrary file read or information disclosure if a client provides `../../` relative paths or absolute system paths.
- **Remediation**: Implement `is_safe_path()` and `validate_safe_path()` to strictly reject any path escaping `data/`, `checkpoints/`, or `experiments/`.

### 4.2 File Upload Validation (Medium Risk)
- **Finding**: While size limit (50MB) and non-empty checks were present in `DatasetService`, validation of MIME types, CSV header structure, and strict `.csv` file extensions was incomplete.
- **Impact**: Attackers could upload non-CSV binary payloads or excessively wide CSV files designed to consume excessive RAM during parsing.
- **Remediation**: Validate file extension, MIME type, and sniff first chunk before parsing. Enforce maximum column and row bounds.

### 4.3 Model Checkpoint Deserialization (Medium Risk)
- **Finding**: `CheckpointManager.load` called `torch.load(..., weights_only=False)` because the serialized bundle includes custom `TabularPreprocessor` objects.
- **Impact**: Loading untrusted `.pt` files could trigger arbitrary code execution via Python pickle vulnerabilities.
- **Remediation**: Confine checkpoint loading to verified internal checkpoint directories; document strict policies against loading external untrusted checkpoints.

### 4.4 CORS & Environment Leakage (Low Risk)
- **Finding**: Default CORS was set to `["*"]` in `api/config.py`.
- **Impact**: Permissive CORS in production environments allows unauthorized web origins to make cross-origin requests to internal endpoints.
- **Remediation**: Make CORS origin list environment-configurable via `CORS_ORIGINS`; disallow wildcard `*` or log prominent security warnings in production mode (`TABSYNTH_ENV=production`).

---

## 5. Reliability & Error Handling Audit

1. **Structured Exception Masking**: The API implements `general_exception_handler`, which logs full tracebacks to the server log while returning a sanitized response (`"An internal error occurred. Request ID: ..."`). This prevents leaking internal filesystem paths to end users.
2. **Concurrency Safety**: `JobManager` utilizes a bounded `ThreadPoolExecutor` and `threading.Semaphore` to throttle background workloads, preventing compute exhaustion on multicore hosts.
3. **Database Concurrency**: SQLite with WAL mode supports concurrent readers while writing. A 30-second busy timeout is configured to prevent `sqlite3.OperationalError: database is locked`.

---

## 6. Action Plan for Phase 07

1. **Security Layer**: Create `api/utils/security.py` with path traversal guards and input sanitizers.
2. **Config Validation**: Harden `api/config.py` with environment variable parsing and strict production assertions.
3. **Startup Recovery**: Add `recover_orphaned_jobs()` in `api/db/session.py`.
4. **Database Operations**: Add `scripts/backup_db.py` and `scripts/restore_db.py`.
5. **Containerization**: Update `Dockerfile` with multi-stage build, non-root user, and optimize `.dockerignore`.
6. **Compose & Health**: Update `docker-compose.yml` with health checks and volume specifications.
7. **Testing & QA**: Add unit tests for security controls, config validation, and database backup; create Phase 07 smoke test.
8. **Documentation**: Provide `DEPLOYMENT_GUIDE.md` and `SECURITY.md`.
