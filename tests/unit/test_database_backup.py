"""Unit tests for online database backup, integrity checking, and restore procedures."""

import os
import sqlite3
import tempfile
import pytest
from scripts.backup_db import create_backup
from scripts.restore_db import restore_database


@pytest.fixture
def sample_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT);")
        conn.execute("INSERT INTO test_table (name) VALUES ('record_1'), ('record_2');")
        conn.commit()

    yield db_path

    if os.path.exists(db_path):
        os.remove(db_path)


def test_online_backup_and_retention(sample_db):
    with tempfile.TemporaryDirectory() as backup_dir:
        # Create first backup
        backup_1 = create_backup(db_path=sample_db, dest_dir=backup_dir, retention=2)
        assert os.path.exists(backup_1)

        # Create second backup
        backup_2 = create_backup(db_path=sample_db, dest_dir=backup_dir, retention=2)
        assert os.path.exists(backup_2)

        # Create third backup (should prune backup_1)
        backup_3 = create_backup(db_path=sample_db, dest_dir=backup_dir, retention=2)
        assert os.path.exists(backup_3)

        # Verify only 2 files remain
        remaining = [f for f in os.listdir(backup_dir) if f.endswith(".db")]
        assert len(remaining) == 2


def test_database_restore(sample_db):
    with tempfile.TemporaryDirectory() as backup_dir:
        backup_file = create_backup(db_path=sample_db, dest_dir=backup_dir)

        # Modify sample_db
        with sqlite3.connect(sample_db) as conn:
            conn.execute("INSERT INTO test_table (name) VALUES ('record_3');")
            conn.commit()

        # Restore from backup
        restore_database(source_backup_path=backup_file, target_db_path=sample_db)

        # Verify restored state has exactly 2 records
        with sqlite3.connect(sample_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM test_table;")
            count = cursor.fetchone()[0]
            assert count == 2
