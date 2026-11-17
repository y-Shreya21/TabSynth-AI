"""Repository classes for CRUD operations on SQLite persistence layer."""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from api.db.session import get_db_connection
from api.db.models import DatasetRecord, ModelRecord, JobRecord, ExperimentRecord, ArtifactRecord


class DatasetRepository:
    """Repository managing dataset records."""

    @staticmethod
    def create(record: DatasetRecord) -> DatasetRecord:
        now = datetime.utcnow().isoformat()
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO datasets (
                    id, name, file_path, num_rows, num_columns,
                    continuous_columns, categorical_columns, cardinalities,
                    target_column, class_distribution, missing_values, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.name,
                    record.file_path,
                    record.num_rows,
                    record.num_columns,
                    json.dumps(record.continuous_columns),
                    json.dumps(record.categorical_columns),
                    json.dumps(record.cardinalities),
                    record.target_column,
                    json.dumps(record.class_distribution) if record.class_distribution else None,
                    json.dumps(record.missing_values) if record.missing_values else None,
                    now,
                ),
            )
        record.created_at = now
        return record

    @staticmethod
    def get_by_id(dataset_id: str) -> Optional[DatasetRecord]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
            if not row:
                return None
            return DatasetRecord(
                id=row["id"],
                name=row["name"],
                file_path=row["file_path"],
                num_rows=row["num_rows"],
                num_columns=row["num_columns"],
                continuous_columns=json.loads(row["continuous_columns"]),
                categorical_columns=json.loads(row["categorical_columns"]),
                cardinalities=json.loads(row["cardinalities"]),
                target_column=row["target_column"],
                class_distribution=json.loads(row["class_distribution"]) if row["class_distribution"] else None,
                missing_values=json.loads(row["missing_values"]) if row["missing_values"] else None,
                created_at=row["created_at"],
            )

    @staticmethod
    def list_all() -> List[DatasetRecord]:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
            return [
                DatasetRecord(
                    id=row["id"],
                    name=row["name"],
                    file_path=row["file_path"],
                    num_rows=row["num_rows"],
                    num_columns=row["num_columns"],
                    continuous_columns=json.loads(row["continuous_columns"]),
                    categorical_columns=json.loads(row["categorical_columns"]),
                    cardinalities=json.loads(row["cardinalities"]),
                    target_column=row["target_column"],
                    class_distribution=json.loads(row["class_distribution"]) if row["class_distribution"] else None,
                    missing_values=json.loads(row["missing_values"]) if row["missing_values"] else None,
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    @staticmethod
    def delete(dataset_id: str) -> bool:
        with get_db_connection() as conn:
            res = conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            return res.rowcount > 0


class ModelRepository:
    """Repository managing trained model records."""

    @staticmethod
    def create(record: ModelRecord) -> ModelRecord:
        now = datetime.utcnow().isoformat()
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO models (
                    id, name, model_type, dataset_id, target_column,
                    checkpoint_path, hyperparameters, metrics, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.name,
                    record.model_type,
                    record.dataset_id,
                    record.target_column,
                    record.checkpoint_path,
                    json.dumps(record.hyperparameters),
                    json.dumps(record.metrics) if record.metrics else None,
                    now,
                ),
            )
        record.created_at = now
        return record

    @staticmethod
    def get_by_id(model_id: str) -> Optional[ModelRecord]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
            if not row:
                return None
            return ModelRecord(
                id=row["id"],
                name=row["name"],
                model_type=row["model_type"],
                dataset_id=row["dataset_id"],
                target_column=row["target_column"],
                checkpoint_path=row["checkpoint_path"],
                hyperparameters=json.loads(row["hyperparameters"]),
                metrics=json.loads(row["metrics"]) if row["metrics"] else None,
                created_at=row["created_at"],
            )

    @staticmethod
    def list_all() -> List[ModelRecord]:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
            return [
                ModelRecord(
                    id=row["id"],
                    name=row["name"],
                    model_type=row["model_type"],
                    dataset_id=row["dataset_id"],
                    target_column=row["target_column"],
                    checkpoint_path=row["checkpoint_path"],
                    hyperparameters=json.loads(row["hyperparameters"]),
                    metrics=json.loads(row["metrics"]) if row["metrics"] else None,
                    created_at=row["created_at"],
                )
                for row in rows
            ]


class JobRepository:
    """Repository managing background jobs and lifecycle tracking."""

    @staticmethod
    def create(record: JobRecord) -> JobRecord:
        now = datetime.utcnow().isoformat()
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, job_type, status, progress, current_step,
                    total_steps, metadata, result, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.job_type,
                    record.status,
                    record.progress,
                    record.current_step,
                    record.total_steps,
                    json.dumps(record.metadata),
                    json.dumps(record.result) if record.result else None,
                    record.error,
                    now,
                    now,
                ),
            )
        record.created_at = now
        record.updated_at = now
        return record

    @staticmethod
    def get_by_id(job_id: str) -> Optional[JobRecord]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return None
            return JobRecord(
                id=row["id"],
                job_type=row["job_type"],
                status=row["status"],
                progress=float(row["progress"]),
                current_step=int(row["current_step"]),
                total_steps=int(row["total_steps"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                result=json.loads(row["result"]) if row["result"] else None,
                error=row["error"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    @staticmethod
    def list_all(limit: int = 50) -> List[JobRecord]:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            return [
                JobRecord(
                    id=row["id"],
                    job_type=row["job_type"],
                    status=row["status"],
                    progress=float(row["progress"]),
                    current_step=int(row["current_step"]),
                    total_steps=int(row["total_steps"]),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    result=json.loads(row["result"]) if row["result"] else None,
                    error=row["error"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

    @staticmethod
    def update_progress(
        job_id: str,
        status: str,
        progress: float,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        now = datetime.utcnow().isoformat()
        with get_db_connection() as conn:
            res = conn.execute(
                """
                UPDATE jobs SET
                    status = ?,
                    progress = ?,
                    current_step = coalesce(?, current_step),
                    total_steps = coalesce(?, total_steps),
                    result = coalesce(?, result),
                    error = coalesce(?, error),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    progress,
                    current_step,
                    total_steps,
                    json.dumps(result) if result is not None else None,
                    error,
                    now,
                    job_id,
                ),
            )
            return res.rowcount > 0


class ExperimentRepository:
    """Repository managing experiment reports and metadata."""

    @staticmethod
    def create(record: ExperimentRecord) -> ExperimentRecord:
        now = datetime.utcnow().isoformat()
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    id, name, description, dataset_id, model_id, metrics, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.name,
                    record.description,
                    record.dataset_id,
                    record.model_id,
                    json.dumps(record.metrics),
                    now,
                ),
            )
        record.created_at = now
        return record

    @staticmethod
    def get_by_id(experiment_id: str) -> Optional[ExperimentRecord]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
            if not row:
                return None
            return ExperimentRecord(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                dataset_id=row["dataset_id"],
                model_id=row["model_id"],
                metrics=json.loads(row["metrics"]) if row["metrics"] else {},
                created_at=row["created_at"],
            )

    @staticmethod
    def list_all() -> List[ExperimentRecord]:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
            return [
                ExperimentRecord(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    dataset_id=row["dataset_id"],
                    model_id=row["model_id"],
                    metrics=json.loads(row["metrics"]) if row["metrics"] else {},
                    created_at=row["created_at"],
                )
                for row in rows
            ]


class ArtifactRepository:
    """Repository managing generated files and artifact references."""

    @staticmethod
    def create(record: ArtifactRecord) -> ArtifactRecord:
        now = datetime.utcnow().isoformat()
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (id, job_id, artifact_type, file_path, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (record.id, record.job_id, record.artifact_type, record.file_path, now),
            )
        record.created_at = now
        return record

    @staticmethod
    def list_by_job_id(job_id: str) -> List[ArtifactRecord]:
        with get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM artifacts WHERE job_id = ?", (job_id,)).fetchall()
            return [
                ArtifactRecord(
                    id=row["id"],
                    job_id=row["job_id"],
                    artifact_type=row["artifact_type"],
                    file_path=row["file_path"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]
