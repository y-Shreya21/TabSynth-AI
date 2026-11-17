# TabSynth Analytics: REST API Reference (v1.0.0)

Base URL: `http://localhost:8000`  
API Prefix: `/api/v1`  
Interactive OpenAPI Documentation: `http://localhost:8000/docs`  
ReDoc Documentation: `http://localhost:8000/redoc`

---

## 1. System & Health Endpoints

### `GET /health`
Returns service health status and detected compute accelerators.
- **Response `200 OK`**:
```json
{
  "status": "healthy",
  "mps_available": true,
  "cuda_available": false,
  "version": "1.0.0"
}
```

### `GET /ready`
Readiness check verifying database connectivity and disk write capabilities.
- **Response `200 OK`**:
```json
{
  "status": "ready",
  "database": "connected",
  "storage_writable": true
}
```

### `GET /version`
Returns software version and service name.
- **Response `200 OK`**:
```json
{
  "version": "1.0.0",
  "service": "TabSynth Analytics API",
  "status": "online"
}
```

---

## 2. Dataset Management Endpoints

### `POST /api/v1/datasets/upload`
Uploads a tabular CSV file, validates non-emptiness, parses headers, detects duplicates, infers schema, and stores metadata in SQLite.
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `file`: CSV file (required)
  - `name`: Dataset alias string (optional)
  - `target_column`: Column name for class guidance (optional)
- **Response `200 OK`**:
```json
{
  "dataset_id": "ds_ca28d2b6a2",
  "name": "Smoke Ingestion Dataset",
  "num_rows": 60,
  "num_columns": 5,
  "target_column": "income",
  "file_path": "data/raw/ds_ca28d2b6a2_smoke_test_data.csv"
}
```

### `GET /api/v1/datasets`
Lists all registered datasets.
- **Response `200 OK`**:
```json
{
  "datasets": [
    {
      "id": "ds_ca28d2b6a2",
      "name": "Adult Income",
      "file_path": "data/raw/adult.csv",
      "num_rows": 32561,
      "num_columns": 15,
      "continuous_columns": ["age", "fnlwgt", "education_num"],
      "categorical_columns": ["workclass", "education", "marital_status"],
      "cardinalities": [9, 16, 7],
      "target_column": "income"
    }
  ]
}
```

### `GET /api/v1/datasets/{dataset_id}`
Returns detailed schema metadata, column statistics, and the top 10 preview records.

### `GET /api/v1/datasets/{dataset_id}/schema`
Returns continuous features, categorical features, cardinalities, and target class distributions.

---

## 3. Model Training & Registry Endpoints

### `POST /api/v1/models/train`
Enqueues an asynchronous training job using the thread-pool job manager.
- **Request Body**:
```json
{
  "dataset_id": "ds_ca28d2b6a2",
  "target_column": "income",
  "model_type": "hybrid_transformer_mlp",
  "model_name": "hybrid_run_01",
  "hyperparameters": {
    "epochs": 30,
    "diffusion_steps": 100,
    "noise_schedule": "cosine",
    "batch_size": 256,
    "learning_rate": 0.0002,
    "d_model": 64,
    "transformer_layers": 2,
    "attention_heads": 4,
    "mlp_hidden_dim": 256,
    "mlp_layers": 3,
    "dropout": 0.1
  },
  "device": "auto"
}
```
- **Response `200 OK`**:
```json
{
  "job_id": "job_9a1b9a8662",
  "status": "queued"
}
```

### `GET /api/v1/models`
Lists all trained model checkpoints and configurations in SQLite.

### `GET /api/v1/models/{model_id}`
Retrieves configuration, training loss, and checkpoint location for a specific model.

---

## 4. Synthetic Data Generation Endpoints

### `POST /api/v1/generate`
Executes reverse diffusion sampling using the trained checkpoint.
- **Request Body**:
```json
{
  "checkpoint_path": "checkpoints/job_9a1b9a8662/model_best.pt",
  "num_samples": 500,
  "condition_strategy": "balanced",
  "cfg_scale": 1.25,
  "steps": 50,
  "output_path": "data/synthetic/generated_records.csv"
}
```
- **Response `200 OK`**:
```json
{
  "status": "completed",
  "num_samples": 500,
  "output_path": "data/synthetic/generated_records.csv",
  "preview_records": [...]
}
```

### `POST /api/v1/generate/conditional`
Synthesizes records targeted strictly at a single class label.
- **Request Body**:
```json
{
  "checkpoint_path": "checkpoints/job_9a1b9a8662/model_best.pt",
  "target_class": 1,
  "num_samples": 300,
  "cfg_scale": 1.5,
  "steps": 50
}
```

---

## 5. Evaluation & Auditing Endpoints

### `POST /api/v1/evaluate/fidelity`
Computes 1D marginal KS distances, Total Variation, Pearson correlation Frobenius distance, and C2ST classifier accuracy.

### `POST /api/v1/evaluate/utility`
Runs downstream Machine Learning utility benchmarks (TSTR vs TRTR) across Logistic Regression and Random Forest classifiers.

### `POST /api/v1/evaluate/privacy`
Audits Distance-to-Closest-Record (DCR) ratios, exact verbatim matches, and shadow Membership Inference Attack (MIA) AUROC.

---

## 6. Background Job Endpoints

### `GET /api/v1/jobs`
Lists recent background tasks with status (`queued`, `running`, `completed`, `failed`, `cancelled`) and progress.

### `GET /api/v1/jobs/{job_id}`
Returns real-time progress and output artifacts for a specific job.

### `POST /api/v1/jobs/{job_id}/cancel`
Signals safe cancellation to an active or queued worker thread.
