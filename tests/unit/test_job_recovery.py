"""Unit tests for orphan job recovery and crash resiliency."""

import os
import sqlite3
import tempfile
import pytest
from api.db.session import SCHEMA_SQL, recover_orphaned_jobs


def test_recover_orphaned_jobs():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        with sqlite3.connect(db_path) as conn:
            conn.executescript(SCHEMA_SQL)
            conn.execute(
                "INSERT INTO jobs (id, job_type, status, progress) VALUES ('j_run', 'train', 'running', 0.4);"
            )
            conn.execute(
                "INSERT INTO jobs (id, job_type, status, progress) VALUES ('j_queue', 'synth', 'queued', 0.0);"
            )
            conn.execute(
                "INSERT INTO jobs (id, job_type, status, progress) VALUES ('j_done', 'train', 'completed', 1.0);"
            )
            conn.execute(
                "INSERT INTO jobs (id, job_type, status, progress) VALUES ('j_cancel', 'synth', 'cancelled', 0.2);"
            )
            conn.commit()

        recovered = recover_orphaned_jobs(db_path)
        assert recovered == 2

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status, error FROM jobs WHERE id = 'j_run';")
            status, error = cursor.fetchone()
            assert status == "failed"
            assert "Server restarted" in error

            cursor.execute("SELECT status FROM jobs WHERE id = 'j_done';")
            assert cursor.fetchone()[0] == "completed"

            cursor.execute("SELECT status FROM jobs WHERE id = 'j_cancel';")
            assert cursor.fetchone()[0] == "cancelled"
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
