"""Phase 06 End-to-End Application & Integration Smoke Test."""

import io
import os
import sys
import time
import pandas as pd
from fastapi.testclient import TestClient

# Ensure root directory in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.config import settings
from api.db.session import init_db
from api.db.repositories import DatasetRepository, ModelRepository, JobRepository, ExperimentRepository
from api.services.dataset_service import DatasetService
from api.services.model_service import ModelService
from api.services.generation_service import GenerationService
from api.services.evaluation_service import EvaluationService
from api.services.job_service import job_manager
from api.schemas.models import TrainRequest, TrainHyperparameters
from api.schemas.generation import GenerateRequest
from api.main import app


def run_phase06_smoke():
    start_time = time.time()
    print("=" * 70)
    print("TabSynth AI - Phase 06 End-to-End Application Smoke Test")
    print("=" * 70)

    # 1. Database Initialization
    print("[*] 1. Initializing SQLite persistence layer...")
    init_db()
    print("[✓] Database initialized at:", settings.db_path)

    # 2. Ingest Test Dataset
    print("[*] 2. Ingesting test tabular dataset...")
    n_rows = 60
    df = pd.DataFrame({
        "age": [float(20 + i) for i in range(n_rows)],
        "capital_gain": [float(i * 100) for i in range(n_rows)],
        "workclass": ["Private" if i % 2 == 0 else "Self-emp" for i in range(n_rows)],
        "education": ["Bachelors" if i % 3 == 0 else "Masters" for i in range(n_rows)],
        "income": [0 if i % 2 == 0 else 1 for i in range(n_rows)],
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    ds_record = DatasetService.upload_dataset(
        file_bytes=csv_bytes,
        filename="smoke_test_data.csv",
        name="Smoke Ingestion Dataset",
        target_column="income",
    )
    print(f"[✓] Ingested dataset '{ds_record.name}' (ID: {ds_record.id}): {ds_record.num_rows} rows × {ds_record.num_columns} cols.")
    assert ds_record.num_rows == n_rows
    assert "age" in ds_record.continuous_columns
    assert "workclass" in ds_record.categorical_columns

    # 3. Model Training Job
    print("[*] 3. Launching asynchronous model training job...")
    train_req = TrainRequest(
        dataset_id=ds_record.id,
        target_column="income",
        model_type="mlp_baseline",
        model_name="smoke_mlp_model",
        hyperparameters=TrainHyperparameters(
            epochs=2,
            diffusion_steps=10,
            batch_size=16,
            d_model=16,
            mlp_hidden_dim=32,
            mlp_layers=2,
            seed=42,
        ),
        device="cpu",
    )

    job_id = ModelService.start_training_job(train_req)
    print(f"[+] Training job enqueued: {job_id}")

    # Poll until completed
    max_wait_seconds = 30
    poll_start = time.time()
    job_completed = False
    while time.time() - poll_start < max_wait_seconds:
        job = job_manager.get_job(job_id)
        if job and job.status == "completed":
            job_completed = True
            break
        elif job and job.status == "failed":
            raise RuntimeError(f"Training job {job_id} failed: {job.error}")
        time.sleep(0.5)

    assert job_completed, f"Training job did not complete within {max_wait_seconds}s"
    print(f"[✓] Training job {job_id} finished with status '{job.status}' (progress {int(job.progress * 100)}%).")

    # 4. Model Registry Inspection
    print("[*] 4. Inspecting model repository...")
    models = ModelRepository.list_all()
    matching_models = [m for m in models if m.dataset_id == ds_record.id]
    assert len(matching_models) > 0, "Model was not registered in ModelRepository"
    trained_model = matching_models[0]
    print(f"[✓] Model registered: '{trained_model.name}' (ID: {trained_model.id}) at {trained_model.checkpoint_path}")
    assert os.path.exists(trained_model.checkpoint_path)

    # 5. Synthetic Generation
    print("[*] 5. Synthesizing records via reverse diffusion...")
    out_csv = "data/synthetic/smoke_synthetic_output.csv"
    gen_req = GenerateRequest(
        checkpoint_path=trained_model.checkpoint_path,
        num_samples=30,
        condition_strategy="balanced",
        cfg_scale=1.0,
        steps=10,
        output_path=out_csv,
    )
    gen_res = GenerationService.generate(gen_req)
    print(f"[✓] Generated {gen_res.num_samples} records. Saved to: {gen_res.output_path}")
    assert os.path.exists(out_csv)
    synth_df = pd.read_csv(out_csv)
    assert len(synth_df) == 30
    assert not synth_df.isnull().any().any(), "Synthetic dataframe contains null values"

    # 6. Statistical Fidelity Evaluation
    print("[*] 6. Running statistical fidelity benchmark...")
    fid_res = EvaluationService.evaluate_fidelity(
        real_data_path=ds_record.file_path,
        synthetic_data_path=out_csv,
        target_column="income",
    )
    print(f"[✓] Fidelity benchmark complete: Mean KS = {fid_res['mean_kolmogorov_smirnov']:.4f}, C2ST = {fid_res['c2st_accuracy']:.4f}")

    # 7. Experiment Registry Inspection
    print("[*] 7. Inspecting experiment registry...")
    exps = ExperimentRepository.list_all()
    assert len(exps) > 0, "No experiments found in database"
    print(f"[✓] Experiment registered in SQLite (Total experiments: {len(exps)}).")

    # 8. REST API Client Verification
    print("[*] 8. Verifying FastAPI endpoints via TestClient...")
    client = TestClient(app)
    r_health = client.get("/health")
    assert r_health.status_code == 200 and r_health.json()["status"] == "healthy"

    r_ready = client.get("/ready")
    assert r_ready.status_code == 200 and r_ready.json()["status"] == "ready"

    r_datasets = client.get("/api/v1/datasets")
    assert r_datasets.status_code == 200 and len(r_datasets.json()["datasets"]) > 0

    r_models = client.get("/api/v1/models")
    assert r_models.status_code == 200 and len(r_models.json()["models"]) > 0

    print("[✓] All REST API endpoints responded with HTTP 200 OK.")

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"[SUCCESS] Phase 06 smoke test passed in {elapsed:.2f}s (< 30s threshold).")
    print("=" * 70)


if __name__ == "__main__":
    run_phase06_smoke()
