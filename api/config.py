"""Application configuration settings for TabSynth Analytics API."""

import os
from typing import List
from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "TabSynth Analytics API"
    app_version: str = "1.0.0"
    app_description: str = "High-Fidelity, Class-Conditional & Privacy-Aware Tabular Diffusion Service"
    
    # Storage and directory locations
    data_dir: str = os.getenv("TABSYNTH_DATA_DIR", "data")
    raw_data_dir: str = os.getenv("TABSYNTH_RAW_DIR", "data/raw")
    synthetic_data_dir: str = os.getenv("TABSYNTH_SYNTHETIC_DIR", "data/synthetic")
    checkpoint_dir: str = os.getenv("TABSYNTH_CHECKPOINT_DIR", "checkpoints")
    experiment_dir: str = os.getenv("TABSYNTH_EXPERIMENT_DIR", "experiments/results")
    
    # Persistence
    db_path: str = os.getenv("TABSYNTH_DB_PATH", "data/tabsynth.db")
    
    # Security and limits
    max_upload_size_bytes: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(50 * 1024 * 1024)))  # 50MB
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])
    
    # Compute and Job Queue
    device: str = os.getenv("TABSYNTH_DEVICE", "auto")
    max_concurrent_jobs: int = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))

    def ensure_directories(self) -> None:
        """Ensure all required runtime directories exist."""
        for path in [
            self.data_dir,
            self.raw_data_dir,
            self.synthetic_data_dir,
            self.checkpoint_dir,
            self.experiment_dir,
            os.path.dirname(self.db_path) or ".",
        ]:
            os.makedirs(path, exist_ok=True)


settings = Settings()
settings.ensure_directories()
