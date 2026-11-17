"""Unit tests for SQLite persistence layer and repository implementations."""

import os
import uuid
import pytest
from api.db.models import DatasetRecord, ModelRecord, JobRecord, ExperimentRecord, ArtifactRecord
from api.db.repositories import (
    DatasetRepository,
    ModelRepository,
    JobRepository,
    ExperimentRepository,
    ArtifactRepository,
)


def test_dataset_repository_lifecycle():
    ds_id = f"test_ds_{uuid.uuid4().hex[:8]}"
    record = DatasetRecord(
        id=ds_id,
        name="Unit Test Dataset",
        file_path="data/raw/test.csv",
        num_rows=100,
        num_columns=5,
        continuous_columns=["col1", "col2"],
        categorical_columns=["col3", "col4"],
        cardinalities=[3, 4],
        target_column="target",
        class_distribution={"0": 0.5, "1": 0.5},
        missing_values={"col1": 0, "col2": 1},
    )

    created = DatasetRepository.create(record)
    assert created.id == ds_id
    assert created.created_at is not None

    fetched = DatasetRepository.get_by_id(ds_id)
    assert fetched is not None
    assert fetched.name == "Unit Test Dataset"
    assert fetched.num_rows == 100
    assert fetched.continuous_columns == ["col1", "col2"]

    all_datasets = DatasetRepository.list_all()
    assert any(d.id == ds_id for d in all_datasets)

    deleted = DatasetRepository.delete(ds_id)
    assert deleted is True
    assert DatasetRepository.get_by_id(ds_id) is None


def test_model_repository_lifecycle():
    model_id = f"test_model_{uuid.uuid4().hex[:8]}"
    record = ModelRecord(
        id=model_id,
        name="Test Hybrid Model",
        model_type="hybrid_transformer_mlp",
        checkpoint_path="checkpoints/test/model.pt",
        hyperparameters={"d_model": 64, "epochs": 5},
        metrics={"best_val_loss": 0.25},
    )

    created = ModelRepository.create(record)
    assert created.id == model_id

    fetched = ModelRepository.get_by_id(model_id)
    assert fetched is not None
    assert fetched.model_type == "hybrid_transformer_mlp"
    assert fetched.hyperparameters["d_model"] == 64

    models = ModelRepository.list_all()
    assert any(m.id == model_id for m in models)


def test_job_repository_progress_and_completion():
    job_id = f"test_job_{uuid.uuid4().hex[:8]}"
    record = JobRecord(
        id=job_id,
        job_type="training",
        status="queued",
        progress=0.0,
        current_step=0,
        total_steps=10,
        metadata={"model": "hybrid"},
    )

    JobRepository.create(record)

    JobRepository.update_progress(
        job_id=job_id,
        status="running",
        progress=0.5,
        current_step=5,
        total_steps=10,
    )

    updated = JobRepository.get_by_id(job_id)
    assert updated is not None
    assert updated.status == "running"
    assert updated.progress == 0.5
    assert updated.current_step == 5

    JobRepository.update_progress(
        job_id=job_id,
        status="completed",
        progress=1.0,
        current_step=10,
        total_steps=10,
        result={"checkpoint": "checkpoints/test.pt"},
    )

    completed = JobRepository.get_by_id(job_id)
    assert completed.status == "completed"
    assert completed.progress == 1.0
    assert completed.result["checkpoint"] == "checkpoints/test.pt"


def test_experiment_and_artifact_repositories():
    exp_id = f"test_exp_{uuid.uuid4().hex[:8]}"
    exp_record = ExperimentRecord(
        id=exp_id,
        name="Unit Test Experiment",
        description="Testing experiment repo",
        metrics={"ks_mean": 0.05, "c2st": 0.52},
    )
    ExperimentRepository.create(exp_record)

    fetched_exp = ExperimentRepository.get_by_id(exp_id)
    assert fetched_exp is not None
    assert fetched_exp.metrics["ks_mean"] == 0.05

    art_id = f"test_art_{uuid.uuid4().hex[:8]}"
    art_record = ArtifactRecord(
        id=art_id,
        job_id="test_job_123",
        artifact_type="synthetic_csv",
        file_path="data/synthetic/test.csv",
    )
    ArtifactRepository.create(art_record)

    artifacts = ArtifactRepository.list_by_job_id("test_job_123")
    assert any(a.id == art_id for a in artifacts)
