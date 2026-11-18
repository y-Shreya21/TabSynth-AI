"""SQLite connection management and database initialization for TabSynth Analytics."""

import sqlite3
import os
from contextlib import contextmanager
from typing import Generator
from api.config import settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    num_rows INTEGER NOT NULL,
    num_columns INTEGER NOT NULL,
    continuous_columns TEXT NOT NULL,
    categorical_columns TEXT NOT NULL,
    cardinalities TEXT NOT NULL,
    target_column TEXT,
    class_distribution TEXT,
    missing_values TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    dataset_id TEXT,
    target_column TEXT,
    checkpoint_path TEXT NOT NULL,
    hyperparameters TEXT NOT NULL,
    metrics TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL DEFAULT 0.0,
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 0,
    metadata TEXT,
    result TEXT,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    dataset_id TEXT,
    model_id TEXT,
    metrics TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id),
    FOREIGN KEY (model_id) REFERENCES models(id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    job_id TEXT,
    artifact_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);
"""


def recover_orphaned_jobs(db_path: str = settings.db_path) -> int:
    """Detects and fails jobs left in 'running' or 'queued' state across server restarts."""
    if not os.path.exists(db_path):
        return 0
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM jobs WHERE status IN ('running', 'queued');")
        stale_jobs = cursor.fetchall()
        if stale_jobs:
            stale_ids = [row[0] for row in stale_jobs]
            cursor.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    error = 'Server restarted or terminated while job was in progress.',
                    updated_at = CURRENT_TIMESTAMP
                WHERE status IN ('running', 'queued');
                """
            )
            conn.commit()
            return len(stale_ids)
    return 0


def init_db(db_path: str = settings.db_path) -> None:
    """Initialize SQLite tables, apply concurrency Pragmas, and recover stale jobs."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    recover_orphaned_jobs(db_path)


@contextmanager
def get_db_connection(db_path: str = settings.db_path) -> Generator[sqlite3.Connection, None, None]:
    """Context manager providing a thread-safe connection with dictionary row access."""
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=30000;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Ensure tables exist on module load
init_db()
