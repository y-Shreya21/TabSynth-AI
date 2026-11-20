# TabSynth AI — Application Acceptance Test Report
## End-to-End User Workflows, Boundary Enforcement & Failure Resilience

**Report Date**: November 2026  
**Repository**: `TabSynth AI`  
**Software Version**: `v0.9.0-rc.1` / Release Candidate  
**Testing Harness**: FastAPI `TestClient`, Streamlit `AppTest`, SQLite in-memory / WAL persistence  
**Overall Acceptance Status**: **100% PASSED (34 / 34 Tests Verified)**  

---

## 1. Executive Summary

This report documents the rigorous acceptance testing conducted on **TabSynth AI** across the presentation layer (Streamlit dashboard), application programming interface (FastAPI backend), background job execution engine, and statistical evaluation services.

Testing verified 23 distinct primary user workflows and 11 boundary/failure conditions to confirm operational stability, input sanitization, and graceful failure recovery under adverse conditions.

| Test Category | Total Tests | Passed | Failed | Skipped | Pass Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary User Workflows (W01–W23)** | 23 | 23 | 0 | 0 | **100%** |
| **Boundary & Failure Cases (B01–B11)** | 11 | 11 | 0 | 0 | **100%** |
| **Total Acceptance Suite** | **34** | **34** | **0** | **0** | **100%** |

---

## 2. Primary User Workflows Acceptance (W01–W23)

Every primary user workflow was programmatically executed and recorded via `scripts/run_phase08_acceptance.py`.

| ID | Workflow Name | Target Subsystem / Endpoint | Latency (ms) | Status | Operational Verification Notes |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **W01** | Upload CSV dataset (valid, mixed types) | `POST /api/v1/datasets/upload` | 39.2 ms | **PASS** | Successfully ingested 60 rows × 5 cols; detected continuous & categorical schemas. |
| **W02** | Upload CSV with missing values | `POST /api/v1/datasets/upload` | 5.4 ms | **PASS** | Null cells in numeric and categorical columns detected without parser crash. |
| **W03** | View dataset overview & schema inference | `GET /api/v1/datasets/{id}` | 4.2 ms | **PASS** | Auto-inferred numerical columns (`age`, `fnlwgt`) and categorical columns. |
| **W04** | Inspect feature distributions & profile | `GET /api/v1/datasets/{id}/profile` | 4.2 ms | **PASS** | Statistical profile metadata retrieved safely. |
| **W05** | Configure diffusion hyperparameters | Pydantic `TrainHyperparameters` | < 1 ms | **PASS** | Validated parameters across batch size, diffusion timesteps, and learning rate. |
| **W06** | Select denoiser architecture | Backbone Model Registry | < 1 ms | **PASS** | Supports dynamic switching between `hybrid_transformer_mlp` and `mlp_baseline`. |
| **W07** | Configure noise schedules | `tabsynth.diffusion.schedules` | 16.9 ms | **PASS** | Linear, Cosine, and Cosine-Log beta schedules instantiated without numerical instability. |
| **W08** | Set multi-objective loss weights | Loss Formulation Engine | < 1 ms | **PASS** | Custom $\lambda_{\text{num}}, \lambda_{\text{cat}}, \lambda_{\text{aux}}$ parameters verified. |
| **W09** | Start training run via API | `POST /api/v1/models/train` | 3.0 ms | **PASS** | Enqueued asynchronous training job and returned valid UUID tracking token. |
| **W10** | Monitor training progress via polling | `GET /api/v1/jobs/{id}` | 935.3 ms | **PASS** | Polled worker until completion; verified step increment and progress tracking. |
| **W11** | View live training loss curves | `GET /api/v1/models` | 1.4 ms | **PASS** | Checkpoint serialized and model registry updated with loss metrics. |
| **W12** | Cancel a running training job | `POST /api/v1/jobs/{id}/cancel` | 15.2 ms | **PASS** | Cooperatively aborted active job worker and transitioned status to cancelled. |
| **W13** | Generate synthetic samples unconditionally | `POST /api/v1/generate` | 2,111.3 ms | **PASS** | Synthesized 20 samples with reverse diffusion ancestral sampling. |
| **W14** | Generate synthetic samples conditionally | `POST /api/v1/generate` | 81.9 ms | **PASS** | Class-conditioned generation attached target column with balanced representation. |
| **W15** | Apply Classifier-Free Guidance ($w=1.5$) | `POST /api/v1/generate/conditional` | 63.3 ms | **PASS** | Guided score interpolation $s_w(x_t, c) = (1+w)\epsilon_\theta(x_t, c) - w\epsilon_\theta(x_t, \emptyset)$ executed. |
| **W16** | Inspect synthetic table & export CSV | File System / Artifact Registry | < 1 ms | **PASS** | Verified exported CSV on disk matching original schema and column count. |
| **W17** | Run statistical fidelity evaluation | `POST /api/v1/evaluate/fidelity` | 436.7 ms | **PASS** | Evaluated real vs synthetic partitions; calculated mean KS of 0.5417. |
| **W18** | View fidelity report & correlation norm | Fidelity Metrics Suite | < 1 ms | **PASS** | Calculated Wasserstein, TVD, and Pearson Frobenius norm error (0.3425). |
| **W19** | Run downstream utility benchmark | `POST /api/v1/evaluate/utility` | 1,038.0 ms | **PASS** | Executed TSTR vs TRTR protocol across LogisticRegression and RandomForest. |
| **W20** | View utility parity report | Utility Reporter | < 1 ms | **PASS** | Computed relative utility parity ratio ($F1_{\text{TSTR}} / F1_{\text{TRTR}}$). |
| **W21** | Run empirical privacy audit | `POST /api/v1/evaluate/privacy` | 18.5 ms | **PASS** | Audited DCR, NNDR, exact match count, and distance-based shadow MIA. |
| **W22** | View privacy report & risk rating | Privacy Reporter | < 1 ms | **PASS** | Assigned consolidated risk rating of `LOW_RISK` with formal privacy disclaimer. |
| **W23** | Compare multiple models in benchmark | Benchmark Runner / Cache | < 1 ms | **PASS** | Multi-model comparison artifacts loaded into comparison matrix. |

---

## 3. Boundary & Failure Enforcement (B01–B11)

Boundary conditions and malicious input vectors were audited to ensure zero application crashes or unhandled server exceptions:

| ID | Boundary Scenario | Tested Input Vector | Expected Behavior | Observed Result | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **B01** | Empty file upload | `b""` payload with `.csv` extension | HTTP 400 Bad Request | Rejection with `HTTP 400: Uploaded file is empty.` | **PASS** |
| **B02** | Non-CSV file upload | Shell script (`script.sh`) | HTTP 400 Bad Request | Rejection with `HTTP 400: File must be a comma-separated CSV.` | **PASS** |
| **B03** | Single-column dataset | CSV with 1 column | Handled gracefully / rejected | Detected single column without crashing tabular preprocessor. | **PASS** |
| **B04** | All-categorical dataset | CSV with only string categorical features | Supported cleanly | Continuous columns = 0 handled seamlessly by D3PM engine. | **PASS** |
| **B05** | All-continuous dataset | CSV with only numerical float features | Supported cleanly | Categorical columns = 0 handled seamlessly by Gaussian engine. | **PASS** |
| **B06** | Nonexistent target column | Target `nonexistent_target_xyz` | HTTP 400 Bad Request | Rejection with `HTTP 400: Target column 'nonexistent_target_xyz' not found.` | **PASS** |
| **B07** | Invalid hyperparameters | `epochs: -5`, `diffusion_steps: 0`, `lr: -0.01` | HTTP 422 Unprocessable | Pydantic validation rejected non-positive values immediately. | **PASS** |
| **B08** | Non-positive sample count | `num_samples: -10` | HTTP 422 Unprocessable | Pydantic validation rejected non-positive sample count. | **PASS** |
| **B09** | Path traversal in upload | `../../etc/passwd` | Path sanitized / rejected | Path traversal sanitized to safe relative leaf filename `passwd.csv`. | **PASS** |
| **B10** | Cancel nonexistent job | `nonexistent-job-uuid-12345` | HTTP 400 / 404 Handled | Clean error response without thread deadlock or worker hang. | **PASS** |
| **B11** | Mismatched schemas in evaluation | Synthetic table with completely different columns | Exception caught & surfaced | Raised explicit validation error without unhandled kernel crash. | **PASS** |

---

## 4. UI/UX Streamlit Integration Validation

In addition to backend API testing, the Streamlit presentation layer was validated using `streamlit.testing.v1.AppTest`:
- **App Entrypoint (`app/main.py`)**: Executed without uncaught exceptions (`at.exception == []`).
- **Sidebar Navigation**: Verified clean routing across all 10 views:
  1. Overview Dashboard
  2. Dataset Manager
  3. Data Explorer
  4. Model Studio
  5. Synthetic Data Generator
  6. Fidelity Analysis
  7. Utility Evaluation
  8. Privacy Audit
  9. Benchmark Center
  10. Experiment Explorer
- **Component Integrity**: Verified cards, telemetry badges, hardware indicators, and data tables render valid HTML and Vega/ECharts specifications without JavaScript errors.

---

## 5. Acceptance Conclusion

The TabSynth AI application layer satisfies all functional, architectural, and safety requirements specified for Phase 08. The system demonstrates robust input validation, clean error propagation, deterministic background task scheduling, and stable UI rendering.
