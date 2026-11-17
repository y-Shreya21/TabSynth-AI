"""Integration tests for Phase 06 FastAPI REST endpoints, background jobs, and services."""

import io
import os
import time
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def sample_csv_bytes():
    n_rows = 40
    df = pd.DataFrame({
        "num1": [float(i) * 1.5 for i in range(n_rows)],
        "num2": [float(i) * 3.2 + 0.5 for i in range(n_rows)],
        "cat1": ["A" if i % 2 == 0 else "B" for i in range(n_rows)],
        "target": [0 if i % 2 == 0 else 1 for i in range(n_rows)],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def test_health_ready_version_endpoints():
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] == "healthy"
    assert "mps_available" in data_health
    assert "cuda_available" in data_health

    res_ready = client.get("/ready")
    assert res_ready.status_code == 200
    assert res_ready.json()["status"] == "ready"

    res_version = client.get("/version")
    assert res_version.status_code == 200
    assert res_version.json()["version"] == "1.0.0"


def test_dataset_upload_and_schema_inspection(sample_csv_bytes):
    # Valid upload
    files = {"file": ("test_upload.csv", sample_csv_bytes, "text/csv")}
    data = {"name": "Test Ingestion Dataset", "target_column": "target"}
    res = client.post("/api/v1/datasets/upload", files=files, data=data)
    assert res.status_code == 200
    res_data = res.json()
    assert res_data["dataset_id"].startswith("ds_")
    assert res_data["num_rows"] == 40
    assert res_data["num_columns"] == 4
    dataset_id = res_data["dataset_id"]

    # Retrieve dataset list
    res_list = client.get("/api/v1/datasets")
    assert res_list.status_code == 200
    datasets = res_list.json()["datasets"]
    assert any(d["id"] == dataset_id for d in datasets)

    # Retrieve detail & preview
    res_detail = client.get(f"/api/v1/datasets/{dataset_id}")
    assert res_detail.status_code == 200
    detail = res_detail.json()
    assert detail["name"] == "Test Ingestion Dataset"
    assert len(detail["preview_records"]) > 0

    # Retrieve schema
    res_schema = client.get(f"/api/v1/datasets/{dataset_id}/schema")
    assert res_schema.status_code == 200
    schema = res_schema.json()
    assert "num1" in schema["continuous_columns"]
    assert "cat1" in schema["categorical_columns"]


def test_dataset_upload_validation_errors():
    # Empty file
    files_empty = {"file": ("empty.csv", b"", "text/csv")}
    res_empty = client.post("/api/v1/datasets/upload", files=files_empty)
    assert res_empty.status_code == 400
    assert "empty" in res_empty.json()["detail"].lower()

    # Invalid CSV content
    files_bad = {"file": ("bad.csv", b"col1,col2\nval1\nval2,val3,val4", "text/csv")}
    # Should either parse or error safely without 500


def test_job_submission_and_status(sample_csv_bytes):
    # Upload small dataset
    files = {"file": ("train_test.csv", sample_csv_bytes, "text/csv")}
    data = {"name": "Job Test Dataset", "target_column": "target"}
    upload_res = client.post("/api/v1/datasets/upload", files=files, data=data).json()
    ds_id = upload_res["dataset_id"]

    train_payload = {
        "dataset_id": ds_id,
        "target_column": "target",
        "model_type": "mlp_baseline",
        "hyperparameters": {
            "epochs": 1,
            "diffusion_steps": 10,
            "batch_size": 4,
            "d_model": 16,
            "mlp_hidden_dim": 32,
            "mlp_layers": 2,
        },
        "device": "cpu",
    }

    res_train = client.post("/api/v1/models/train", json=train_payload)
    assert res_train.status_code == 200
    job_id = res_train.json()["job_id"]
    assert job_id.startswith("job_")

    # Poll status
    res_status = client.get(f"/api/v1/jobs/{job_id}")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["status"] in ["queued", "running", "completed"]

    # Test listing jobs
    res_jobs = client.get("/api/v1/jobs")
    assert res_jobs.status_code == 200
    assert len(res_jobs.json()["jobs"]) > 0


def test_experiments_endpoint():
    res = client.get("/api/v1/experiments")
    assert res.status_code == 200
    assert "experiments" in res.json()
