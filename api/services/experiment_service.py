"""Service managing experiment results, comparisons, and historical tracking."""

import os
import json
import logging
from typing import Dict, Any, List
from fastapi import HTTPException, status
from api.config import settings
from api.db.models import ExperimentRecord
from api.db.repositories import ExperimentRepository

logger = logging.getLogger("tabsynth.experiments")


class ExperimentService:
    """Manages experiment records across database and filesystem artifacts."""

    @staticmethod
    def list_experiments() -> List[Dict[str, Any]]:
        """List experiments stored in SQLite and serialized in experiments/results."""
        records = ExperimentRepository.list_all()
        result = [r.model_dump() for r in records]

        # Scan filesystem for Phase 05 serialized benchmark logs
        if os.path.exists(settings.experiment_dir):
            for fname in os.listdir(settings.experiment_dir):
                if fname.endswith(".json") and not any(r["id"] == fname for r in result):
                    fpath = os.path.join(settings.experiment_dir, fname)
                    result.append({
                        "id": fname.replace(".json", ""),
                        "name": fname.replace(".json", "").replace("_", " ").title(),
                        "description": "Serialized benchmark results from filesystem",
                        "metrics_file": fpath,
                    })

        return result

    @staticmethod
    def get_experiment(experiment_id: str) -> Dict[str, Any]:
        """Retrieve detailed experiment metrics and configuration."""
        record = ExperimentRepository.get_by_id(experiment_id)
        if record:
            return record.model_dump()

        # Check filesystem
        fpath = os.path.join(settings.experiment_dir, f"{experiment_id}.json")
        if not os.path.exists(fpath):
            fpath = os.path.join(settings.experiment_dir, experiment_id)
        
        if os.path.exists(fpath) and os.path.isfile(fpath):
            with open(fpath, "r") as f:
                data = json.load(f)
            return {
                "id": experiment_id,
                "name": experiment_id.replace("_", " ").title(),
                "metrics": data,
            }

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' was not found.",
        )
