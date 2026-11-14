# TabSynth AI: REST API Specification (FastAPI)
## TabSynth Analytics Engine Backend OpenAPI Contract

---

**Version:** 1.0.0  
**Protocol:** HTTP/1.1 REST + JSON / Multipart Form  
**Base URL:** `http://localhost:8000/api/v1`  
**API Documentation (Swagger UI):** `http://localhost:8000/docs`  
**API Documentation (ReDoc):** `http://localhost:8000/redoc`  

---

## 1. Overview & Architectural Principles

The TabSynth AI REST API serves as the programmatic interface for:
- Uploading and validating raw tabular datasets.
- Automatic feature schema profiling and zero-leakage preprocessing.
- Asynchronous model training and checkpoint orchestration.
- High-performance class-conditional synthetic data generation.
- Automated 3D evaluation (Statistical Fidelity, Downstream ML Utility, and Empirical Privacy Audits).

All endpoints follow RESTful conventions, returning standard HTTP response codes and typed JSON responses validated by Pydantic v2.

---

## 2. API Endpoints Summary Matrix

| Method | Endpoint | Description | Request Body | Response Status |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/datasets/upload` | Upload raw CSV dataset | `multipart/form-data` | `201 Created` |
| `GET` | `/datasets/schema` | Retrieve inferred dataset schema | Query: `dataset_id` | `200 OK` |
| `POST` | `/models/train` | Trigger asynchronous diffusion training | `TrainRequest` (JSON) | `202 Accepted` |
| `GET` | `/models/status` | Query training progress and loss | Query: `job_id` | `200 OK` |
| `POST` | `/generate` | Generate class-conditional synthetic data | `GenerateRequest` (JSON)| `200 OK` |
| `POST` | `/evaluate/fidelity` | Run statistical fidelity benchmarks | `FidelityEvalRequest` | `200 OK` |
| `POST` | `/evaluate/utility` | Run TSTR vs TRTR downstream utility | `UtilityEvalRequest` | `200 OK` |
| `POST` | `/evaluate/privacy` | Run DCR, NNDR, and MIA privacy audit | `PrivacyEvalRequest` | `200 OK` |
| `GET` | `/experiments` | List historical experimental runs | Query: `limit`, `offset` | `200 OK` |

---

## 3. Detailed Endpoint Contracts & Schemas

### 3.1 POST `/datasets/upload`
Uploads a raw CSV file, triggers automatic continuous vs categorical column detection, generates a unique `dataset_id`, and stores immutable copies in `data/raw/`.

#### Request:
- **Content-Type:** `multipart/form-data`
- **Form Parameters:**
  - `file`: Raw CSV file binary.
  - `name`: String identifier (e.g., `"adult_income"`).
  - `target_column`: Optional target attribute name (e.g., `"income"`).

#### Response (`201 Created`):
```json
{
  "dataset_id": "ds_a89f3b20",
  "name": "adult_income",
  "num_rows": 48842,
  "num_columns": 15,
  "target_column": "income",
  "uploaded_at": "2026-11-14T15:00:00Z",
  "file_size_bytes": 3974120
}
```

---

### 3.2 GET `/datasets/schema`
Retrieves the inferred schema, column categories, continuous bounds, and missingness metrics.

#### Request:
- **Query Parameter:** `dataset_id=ds_a89f3b20`

#### Response (`200 OK`):
```json
{
  "dataset_id": "ds_a89f3b20",
  "continuous_columns": [
    {"name": "age", "min": 17.0, "max": 90.0, "median": 37.0, "missing_count": 0},
    {"name": "fnlwgt", "min": 12285.0, "max": 1484705.0, "median": 178356.0, "missing_count": 0},
    {"name": "education-num", "min": 1.0, "max": 16.0, "median": 10.0, "missing_count": 0},
    {"name": "capital-gain", "min": 0.0, "max": 99999.0, "median": 0.0, "missing_count": 0},
    {"name": "capital-loss", "min": 0.0, "max": 4356.0, "median": 0.0, "missing_count": 0},
    {"name": "hours-per-week", "min": 1.0, "max": 99.0, "median": 40.0, "missing_count": 0}
  ],
  "categorical_columns": [
    {"name": "workclass", "cardinality": 9, "categories": ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Self-emp-inc", "Federal-gov", "Without-pay", "Never-worked", "<missing>"]},
    {"name": "education", "cardinality": 16},
    {"name": "marital-status", "cardinality": 7},
    {"name": "occupation", "cardinality": 15},
    {"name": "relationship", "cardinality": 6},
    {"name": "race", "cardinality": 5},
    {"name": "sex", "cardinality": 2},
    {"name": "native-country", "cardinality": 42}
  ],
  "target_column": {
    "name": "income",
    "type": "categorical",
    "cardinality": 2,
    "class_distribution": {
      "<=50K": 0.7607,
      ">50K": 0.2393
    }
  }
}
```

---

### 3.3 POST `/models/train`
Dispatches a background worker job to train the TabSynth Hybrid Diffusion model (or MLP baseline) on the preprocessed training partition.

#### Request Body (`TrainRequest` JSON):
```json
{
  "dataset_id": "ds_a89f3b20",
  "model_type": "hybrid_transformer_mlp",
  "hyperparameters": {
    "d_model": 64,
    "transformer_layers": 2,
    "attention_heads": 4,
    "mlp_hidden_dim": 256,
    "dropout": 0.10,
    "diffusion_steps": 1000,
    "noise_schedule": "cosine",
    "batch_size": 256,
    "learning_rate": 0.0002,
    "epochs": 150,
    "lambda_num": 1.0,
    "lambda_cat": 0.2,
    "lambda_aux": 0.1,
    "p_uncond": 0.15
  },
  "device": "mps",
  "seed": 42
}
```

#### Response (`202 Accepted`):
```json
{
  "job_id": "job_tr_7719b2",
  "status": "queued",
  "created_at": "2026-11-14T15:05:00Z",
  "message": "Training job queued on Apple Silicon MPS device."
}
```

---

### 3.4 GET `/models/status`
Polls the execution state of an ongoing training job.

#### Response (`200 OK`):
```json
{
  "job_id": "job_tr_7719b2",
  "status": "running",
  "current_epoch": 45,
  "total_epochs": 150,
  "progress_percentage": 30.0,
  "elapsed_seconds": 182.4,
  "estimated_remaining_seconds": 425.6,
  "metrics": {
    "train_loss_total": 0.4215,
    "loss_num_mse": 0.2810,
    "loss_cat_kl": 0.1205,
    "loss_aux_ce": 0.0200,
    "val_loss_total": 0.4350
  },
  "best_checkpoint_path": "checkpoints/tabsynth_adult_income_best.pt"
}
```

---

### 3.5 POST `/generate`
Executes reverse diffusion sampling conditioned on user-specified class distributions and guidance scales.

#### Request Body (`GenerateRequest` JSON):
```json
{
  "checkpoint_id": "tabsynth_adult_income_best.pt",
  "num_samples": 10000,
  "condition_mode": "balanced",
  "target_column": "income",
  "target_distribution": {
    "<=50K": 0.50,
    ">50K": 0.50
  },
  "classifier_free_guidance_scale": 1.25,
  "reverse_steps": 1000,
  "device": "mps",
  "export_format": "csv"
}
```

#### Response (`200 OK`):
```json
{
  "generation_id": "gen_88a91c",
  "status": "completed",
  "num_samples_generated": 10000,
  "elapsed_sampling_seconds": 128.5,
  "download_url": "/api/v1/downloads/gen_88a91c.csv",
  "preview_records": [
    {
      "age": 42,
      "workclass": "Private",
      "fnlwgt": 184520,
      "education": "Bachelors",
      "education-num": 13,
      "marital-status": "Married-civ-spouse",
      "occupation": "Exec-managerial",
      "relationship": "Husband",
      "race": "White",
      "sex": "Male",
      "capital-gain": 5178,
      "capital-loss": 0,
      "hours-per-week": 45,
      "native-country": "United-States",
      "income": ">50K"
    }
  ]
}
```

---

### 3.6 POST `/evaluate/fidelity`
Executes automated statistical fidelity comparison between generated synthetic data and the held-out real test set.

#### Response (`200 OK`):
```json
{
  "dataset_name": "Adult Income",
  "num_synthetic_records": 10000,
  "fidelity_metrics": {
    "mean_kolmogorov_smirnov": 0.0384,
    "mean_wasserstein_distance": 0.0412,
    "mean_total_variation_distance": 0.0211,
    "correlation_frobenius_distance": 0.0982,
    "c2st_discriminator_accuracy": 0.5310,
    "sdmetrics_quality_score": 0.9420
  },
  "column_level_breakdown": {
    "age": {"ks_stat": 0.0251, "p_value": 0.421},
    "capital-gain": {"ks_stat": 0.0412, "p_value": 0.180},
    "workclass": {"tvd": 0.0185},
    "education": {"tvd": 0.0210}
  }
}
```

---

### 3.7 POST `/evaluate/utility`
Executes TSTR vs TRTR downstream classification benchmarking across Logistic Regression, Random Forest, and XGBoost on untouched real $D_{\text{test}}$.

#### Response (`200 OK`):
```json
{
  "dataset_name": "Adult Income",
  "target_column": "income",
  "minority_class_label": ">50K",
  "classifiers": {
    "xgboost": {
      "trtr": {"accuracy": 0.8715, "macro_f1": 0.8120, "minority_recall": 0.6520, "auroc": 0.9250},
      "tstr": {"accuracy": 0.8650, "macro_f1": 0.8065, "minority_recall": 0.7310, "auroc": 0.9180},
      "relative_utility_parity": 0.9932,
      "minority_recall_delta": "+0.0790 (+12.1% relative boost)"
    },
    "random_forest": {
      "trtr": {"accuracy": 0.8540, "macro_f1": 0.7890, "minority_recall": 0.6180, "auroc": 0.9080},
      "tstr": {"accuracy": 0.8490, "macro_f1": 0.7820, "minority_recall": 0.6840, "auroc": 0.9010},
      "relative_utility_parity": 0.9911
    },
    "logistic_regression": {
      "trtr": {"accuracy": 0.8240, "macro_f1": 0.7410, "minority_recall": 0.5420, "auroc": 0.8750},
      "tstr": {"accuracy": 0.8210, "macro_f1": 0.7380, "minority_recall": 0.5890, "auroc": 0.8710},
      "relative_utility_parity": 0.9960
    }
  }
}
```

---

### 3.8 POST `/evaluate/privacy`
Runs empirical privacy auditing including DCR ratio, NNDR, and shadow-model MIA.

#### Response (`200 OK`):
```json
{
  "sample_count": 10000,
  "dcr_analysis": {
    "mean_dcr_train": 0.2841,
    "mean_dcr_test": 0.2855,
    "dcr_ratio_train_over_test": 0.9951,
    "5th_percentile_dcr_train": 0.1420,
    "exact_training_reproductions": 0,
    "exact_reproduction_rate": 0.0000
  },
  "nndr_analysis": {
    "mean_nndr": 0.8240,
    "nndr_below_0_5_percentage": 0.0012
  },
  "membership_inference_attack": {
    "shadow_model_mia_auroc": 0.5210,
    "attack_advantage": 0.0420,
    "risk_level": "LOW_EMPIRICAL_RISK"
  },
  "disclaimer": "Diagnostic privacy metrics evaluate heuristic memorization and do not guarantee mathematical differential privacy."
}
```

---

## 4. Error Codes & Exception Handling

The API adheres to standard HTTP status codes:
- **`400 Bad Request`:** Validation failure in request payload or incompatible dataset schema.
- **`404 Not Found`:** Specified `dataset_id` or model checkpoint does not exist.
- **`422 Unprocessable Entity`:** Pydantic schema validation error (e.g., negative sample counts or guidance scale $< 0$).
- **`500 Internal Server Error`:** Unhandled runtime exception (logged with stack trace; sanitized user response).
- **`503 Service Unavailable`:** Hardware device busy or insufficient memory available.
