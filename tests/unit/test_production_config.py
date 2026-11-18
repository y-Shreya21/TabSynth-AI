"""Unit tests for production configuration, environment parsing, and CORS rules."""

import os
import pytest
from api.config import Settings, _parse_cors_origins


def test_cors_origin_parsing(monkeypatch):
    # Comma-separated
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000, https://app.example.com")
    origins = _parse_cors_origins()
    assert origins == ["http://localhost:3000", "https://app.example.com"]

    # JSON formatted
    monkeypatch.setenv("CORS_ORIGINS", '["http://localhost:8501", "http://127.0.0.1:8501"]')
    origins_json = _parse_cors_origins()
    assert origins_json == ["http://localhost:8501", "http://127.0.0.1:8501"]

    # Default wildcard when empty
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert _parse_cors_origins() == ["*"]


def test_settings_development_defaults(monkeypatch):
    monkeypatch.setenv("TABSYNTH_ENV", "development")
    s = Settings()
    assert s.env == "development"
    assert s.device == "auto" or os.getenv("TABSYNTH_DEVICE", "auto")
    roots = s.get_allowed_storage_roots()
    assert len(roots) >= 4
    s.validate_production_config()


def test_settings_production_wildcard_warning(monkeypatch, caplog):
    monkeypatch.setenv("TABSYNTH_ENV", "production")
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    s = Settings()
    assert s.env == "production"
    s.validate_production_config()
    assert any("SECURITY WARNING" in record.message for record in caplog.records)
