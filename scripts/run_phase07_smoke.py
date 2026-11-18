#!/usr/bin/env python3
"""Phase 07 Production Hardening, Security & Deployment Smoke Test.

Verifies:
1. Environment & configuration validation.
2. Path traversal security protection.
3. Upload sanitization and file extension enforcement.
4. Orphan job recovery sweep across restarts.
5. SQLite online backup and integrity check.
6. REST API health, readiness, and security endpoint rejection.
"""

import os
import sys
import tempfile
import sqlite3
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.config import settings
from api.server import app
from api.db.session import SCHEMA_SQL, recover_orphaned_jobs
from api.utils.security import (
    is_safe_path,
    validate_safe_path,
    sanitize_filename,
    validate_csv_upload,
)
from scripts.backup_db import create_backup


def test_configuration_sanitization():
    print("[*] 1. Validating runtime configuration and storage roots...")
    assert settings.env in ["development", "staging", "production"], f"Invalid env: {settings.env}"
    roots = settings.get_allowed_storage_roots()
    assert len(roots) >= 4, "Insufficient storage roots registered."
    for r in roots:
        assert os.path.isabs(os.path.abspath(r)), f"Storage root {r} must resolve to absolute path."
    print("    [✓] Configuration and storage roots verified.")


def test_path_traversal_guards():
    print("[*] 2. Verifying path traversal and confinement guards...")
    allowed_roots = [settings.data_dir, settings.checkpoint_dir]

    # Malicious traversal vectors
    malicious_paths = [
        "../../etc/passwd",
        "/etc/shadow",
        "data/../../secret.key",
        "data/raw/../../../root",
        "\x00data/raw/test.csv",
    ]
    for p in malicious_paths:
        assert not is_safe_path(p, allowed_roots), f"Path should have been rejected as unsafe: {p}"
        with pytest.raises(HTTPException) as exc_info:
            validate_safe_path(p, allowed_roots)
        assert exc_info.value.status_code == 400, f"Expected 400 for unsafe path {p}"

    # Legitimate paths
    safe_data_file = os.path.join(settings.data_dir, "raw", "sample.csv")
    assert is_safe_path(safe_data_file, allowed_roots), f"Valid path was rejected: {safe_data_file}"
    validated = validate_safe_path(safe_data_file, allowed_roots)
    assert os.path.isabs(validated), "Validated path must be absolute canonical."
    print("    [✓] All path traversal attempts successfully blocked.")


def test_upload_sanitization():
    print("[*] 3. Testing upload file sanitization & binary rejection...")
    # Safe filename sanitization
    assert sanitize_filename("../../../malicious_name.csv") == "malicious_name.csv"
    assert sanitize_filename("bad:name?.csv") == "bad_name_.csv"
    assert sanitize_filename("auto_ext").endswith(".csv")

    # Extension rejection
    with pytest.raises(HTTPException) as exc:
        validate_csv_upload("script.sh", b"echo 'hack'")
    assert exc.value.status_code == 400

    # Binary sniffing rejection
    with pytest.raises(HTTPException) as exc:
        validate_csv_upload("payload.csv", b"\x00\x01\x02binarydata")
    assert exc.value.status_code == 400

    # Empty file rejection
    with pytest.raises(HTTPException) as exc:
        validate_csv_upload("empty.csv", b"")
    assert exc.value.status_code == 400

    # Valid CSV accept
    validate_csv_upload("valid.csv", b"col1,col2\n1,2\n3,4\n")
    print("    [✓] Upload validation controls passed.")


def test_orphan_job_recovery():
    print("[*] 4. Testing orphan job recovery across server restarts...")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        tmp_db_path = tmp_db.name

    try:
        with sqlite3.connect(tmp_db_path) as conn:
            conn.executescript(SCHEMA_SQL)
            conn.execute(
                "INSERT INTO jobs (id, job_type, status, progress) VALUES ('job_stale_1', 'training', 'running', 0.5);"
            )
            conn.execute(
                "INSERT INTO jobs (id, job_type, status, progress) VALUES ('job_stale_2', 'synthesis', 'queued', 0.0);"
            )
            conn.execute(
                "INSERT INTO jobs (id, job_type, status, progress) VALUES ('job_good', 'training', 'completed', 1.0);"
            )
            conn.commit()

        recovered_count = recover_orphaned_jobs(tmp_db_path)
        assert recovered_count == 2, f"Expected 2 recovered jobs, got {recovered_count}"

        with sqlite3.connect(tmp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, status, error FROM jobs WHERE id = 'job_stale_1';")
            row1 = cursor.fetchone()
            assert row1[1] == "failed", f"Expected failed status, got {row1[1]}"
            assert "Server restarted" in row1[2]

            cursor.execute("SELECT id, status FROM jobs WHERE id = 'job_good';")
            row_good = cursor.fetchone()
            assert row_good[1] == "completed", "Completed job was erroneously modified."

        print(f"    [✓] Successfully recovered {recovered_count} orphan jobs to 'failed' state.")
    finally:
        if os.path.exists(tmp_db_path):
            os.remove(tmp_db_path)


def test_sqlite_online_backup():
    print("[*] 5. Running online SQLite database backup and verification...")
    with tempfile.TemporaryDirectory() as tmp_backup_dir:
        backup_path = create_backup(
            db_path=settings.db_path,
            dest_dir=tmp_backup_dir,
            retention=3,
        )
        assert os.path.exists(backup_path), "Backup archive was not created."
        assert os.path.getsize(backup_path) > 0, "Backup archive is empty."
        print(f"    [✓] Verified online backup: '{os.path.basename(backup_path)}'.")


def test_api_health_and_security_endpoints():
    print("[*] 6. Testing FastAPI security controls & probes via TestClient...")
    client = TestClient(app)

    # Health & readiness
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "healthy"

    resp_ready = client.get("/ready")
    assert resp_ready.status_code == 200
    assert resp_ready.json()["status"] == "ready"

    # Blocked traversal on /datasets/schema
    resp_traversal = client.get("/api/v1/datasets/schema?data_path=../../etc/passwd")
    assert resp_traversal.status_code == 400
    assert "Access denied" in resp_traversal.json()["detail"]

    # Blocked traversal on evaluation endpoints
    resp_fid_traversal = client.post(
        "/api/v1/evaluate/fidelity",
        json={"real_data_path": "../../../etc/passwd", "synthetic_data_path": "data/synthetic/adult_synthetic_test.csv"},
    )
    assert resp_fid_traversal.status_code == 400

    print("    [✓] All API security barriers and health probes passed.")


def main():
    print("=" * 70)
    print("TabSynth AI - Phase 07 Production Hardening & Security Smoke Test")
    print("=" * 70)

    test_configuration_sanitization()
    test_path_traversal_guards()
    test_upload_sanitization()
    test_orphan_job_recovery()
    test_sqlite_online_backup()
    test_api_health_and_security_endpoints()

    print("=" * 70)
    print("[SUCCESS] All Phase 07 production hardening checks passed cleanly!")
    print("=" * 70)


if __name__ == "__main__":
    main()
