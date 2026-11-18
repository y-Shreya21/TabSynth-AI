"""Application configuration settings for TabSynth Analytics API."""

import os
import json
import logging
from typing import List
from pydantic import BaseModel, Field

logger = logging.getLogger("tabsynth.config")


def _parse_cors_origins() -> List[str]:
    """Parse CORS origins from environment variable or return development default."""
    raw = os.getenv("CORS_ORIGINS", "")
    if not raw:
        return ["*"]
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


class Settings(BaseModel):
    # Application Metadata
    app_name: str = "TabSynth Analytics API"
    app_version: str = "1.0.0"
    app_description: str = "High-Fidelity, Class-Conditional & Privacy-Aware Tabular Diffusion Service"
    env: str = Field(default_factory=lambda: os.getenv("TABSYNTH_ENV", "development").lower())
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())

    # Host and Networking
    api_host: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    api_workers: int = Field(default_factory=lambda: int(os.getenv("API_WORKERS", "1")))
    api_url: str = Field(default_factory=lambda: os.getenv("TABSYNTH_API_URL", "http://127.0.0.1:8000"))

    # Streamlit Frontend Settings
    streamlit_port: int = Field(default_factory=lambda: int(os.getenv("STREAMLIT_SERVER_PORT", "8501")))
    streamlit_headless: bool = Field(default_factory=lambda: os.getenv("STREAMLIT_SERVER_HEADLESS", "true").lower() == "true")

    # Storage and directory locations
    data_dir: str = Field(default_factory=lambda: os.getenv("TABSYNTH_DATA_DIR", "data"))
    raw_data_dir: str = Field(default_factory=lambda: os.getenv("TABSYNTH_RAW_DIR", "data/raw"))
    synthetic_data_dir: str = Field(default_factory=lambda: os.getenv("TABSYNTH_SYNTHETIC_DIR", "data/synthetic"))
    checkpoint_dir: str = Field(default_factory=lambda: os.getenv("TABSYNTH_CHECKPOINT_DIR", "checkpoints"))
    experiment_dir: str = Field(default_factory=lambda: os.getenv("TABSYNTH_EXPERIMENT_DIR", "experiments/results"))
    backup_dir: str = Field(default_factory=lambda: os.getenv("TABSYNTH_BACKUP_DIR", "backups"))

    # Persistence
    db_path: str = Field(default_factory=lambda: os.getenv("TABSYNTH_DB_PATH", "data/tabsynth.db"))

    # Security and limits
    max_upload_size_bytes: int = Field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(50 * 1024 * 1024)))  # 50MB
    )
    allowed_origins: List[str] = Field(default_factory=_parse_cors_origins)

    # Compute and Job Queue
    device: str = Field(default_factory=lambda: os.getenv("TABSYNTH_DEVICE", "auto"))
    max_concurrent_jobs: int = Field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_JOBS", "1")))

    def ensure_directories(self) -> None:
        """Ensure all required runtime directories exist."""
        for path in [
            self.data_dir,
            self.raw_data_dir,
            self.synthetic_data_dir,
            self.checkpoint_dir,
            self.experiment_dir,
            self.backup_dir,
            os.path.dirname(self.db_path) or ".",
        ]:
            os.makedirs(path, exist_ok=True)

    def get_allowed_storage_roots(self) -> List[str]:
        """Returns list of canonical directory roots for safe path confinement."""
        return [
            self.data_dir,
            self.raw_data_dir,
            self.synthetic_data_dir,
            self.checkpoint_dir,
            self.experiment_dir,
            self.backup_dir,
        ]

    def validate_production_config(self) -> None:
        """Validates configuration sanity and issues warnings or errors for production."""
        self.ensure_directories()
        
        # Verify writable storage
        for directory in [self.data_dir, self.checkpoint_dir, self.experiment_dir]:
            if not os.access(directory, os.W_OK):
                logger.error(f"Directory '{directory}' is not writable!")
                if self.env == "production":
                    raise PermissionError(f"Production storage directory '{directory}' is not writable.")

        # Warn on insecure CORS in production
        if self.env == "production" and "*" in self.allowed_origins:
            logger.warning(
                "SECURITY WARNING: Wildcard CORS ('*') is configured in production environment. "
                "Specify explicit domains using CORS_ORIGINS."
            )


settings = Settings()
settings.ensure_directories()
settings.validate_production_config()
