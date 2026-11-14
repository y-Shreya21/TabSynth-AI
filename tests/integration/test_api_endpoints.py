"""Integration tests for FastAPI REST API endpoints."""

import os
import pytest

try:
    from fastapi.testclient import TestClient
    from api.server import app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI or TestClient not installed")


@pytest.fixture
def client():
    if not HAS_FASTAPI:
        pytest.skip("FastAPI not installed")
    return TestClient(app)


def test_root_and_health_endpoints(client):
    """Verify root discovery and healthcheck endpoints."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "online"

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"
    assert "mps_available" in res_health.json()


def test_get_dataset_schema(client):
    """Verify schema inference endpoint against local dataset."""
    if not os.path.exists("data/raw/adult.csv"):
        pytest.skip("Adult Income dataset not downloaded yet.")

    res = client.get("/api/v1/datasets/schema?data_path=data/raw/adult.csv&target_column=income")
    assert res.status_code == 200
    data = res.json()
    assert data["num_rows"] > 0
    assert "age" in data["continuous_columns"]
    assert "education" in data["categorical_columns"]
    assert data["class_distribution"] is not None


def test_fidelity_evaluation_endpoint(client):
    """Verify statistical fidelity evaluation API endpoint."""
    real_path = "data/raw/adult.csv"
    synth_path = "data/synthetic/adult_synthetic_test.csv"

    if not os.path.exists(real_path) or not os.path.exists(synth_path):
        pytest.skip("Data files for fidelity test not present.")

    payload = {
        "real_data_path": real_path,
        "synthetic_data_path": synth_path,
        "target_column": "income",
    }
    res = client.post("/api/v1/evaluate/fidelity", json=payload)
    assert res.status_code == 200
    metrics = res.json()
    assert "mean_kolmogorov_smirnov" in metrics
    assert "c2st_accuracy" in metrics


def test_list_experiments_endpoint(client):
    """Verify experiments listing endpoint."""
    res = client.get("/api/v1/experiments")
    assert res.status_code == 200
    assert "experiments" in res.json()
