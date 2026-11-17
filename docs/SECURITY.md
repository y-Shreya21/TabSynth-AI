# TabSynth AI Security & Threat Model

This document outlines the security architecture, threat model, data privacy safeguards, and vulnerability reporting procedures for the TabSynth AI application.

---

## 1. Threat Model & Defense In Depth

### 1.1 Ingestion Security
- **File Upload Limits**: Enforces a strict upload size limit (default 50MB, configurable via `MAX_UPLOAD_SIZE_BYTES`) to prevent denial-of-service via resource exhaustion.
- **Path Traversal Defense**: All uploaded filenames are stripped of relative path components (`../`, `..\`) and sanitized with regular expression filters allowing only alphanumeric, dash, underscore, and dot characters.
- **CSV Format Validation**: Uploaded files are checked for non-emptiness, byte integrity, and duplicate column names at header parsing time.

### 1.2 Execution & Compute Isolation
- **Concurrency Limiting**: Background training and sampling tasks are bounded by a concurrency semaphore to prevent memory exhaustion and crashes on accelerator hardware (MPS / CUDA).
- **Graceful Error Sanitization**: API exception handlers intercept internal Python exceptions and return structured JSON error payloads with correlation Request IDs, preventing internal code stack traces from leaking to clients.
- **Zero Raw Record Logging**: Structured request logging records HTTP routes, response codes, and latency, but never writes user dataset rows or sensitive payloads into application log streams.

### 1.3 Container & Host Security
- **Non-Root Runtime User**: The Docker container creates and executes under dedicated user `tabsynth` (UID 1000).
- **Isolated Storage Volumes**: Host directories mounted into containers are restricted to explicit application partitions (`/app/data`, `/app/checkpoints`, `/app/experiments/results`).

---

## 2. Formal Non-Differential Privacy Disclaimer

> [!CAUTION]
> **Academic Research & Privacy Scope:**
> Empirical privacy metrics implemented in TabSynth AI (Distance-to-Closest-Record, Nearest Neighbor Distance Ratios, Membership Inference Attack ROC-AUC, Attribute Inference Advantage, and Singling-Out Risk) evaluate empirical memorization and reconstruction vulnerability on specific finite sample partitions.  
> 
> These empirical diagnostics **DO NOT** provide provable, worst-case mathematical Differential Privacy guarantees $(\epsilon, \delta)$. Provable mathematical guarantees require formal differential privacy mechanisms (such as DP-SGD with calibrated Gaussian noise addition and per-sample gradient clipping) during model training.

---

## 3. Vulnerability Reporting

To report security issues or private data disclosures, please email `security@tabsynth-ai.org` or open a private advisory on GitHub.
