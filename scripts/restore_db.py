#!/usr/bin/env python3
"""Database restore utility with pre-flight integrity verification and safety snapshots."""

import os
import sys
import shutil
import sqlite3
import argparse
import logging
import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.config import settings

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("tabsynth.restore")


def restore_database(
    source_backup_path: str,
    target_db_path: str = settings.db_path,
) -> None:
    """Restores SQLite database from a verified backup archive."""
    if not os.path.exists(source_backup_path):
        raise FileNotFoundError(f"Source backup file not found at: '{source_backup_path}'")

    logger.info(f"Step 1: Verifying integrity of backup file '{source_backup_path}'...")
    check_conn = sqlite3.connect(source_backup_path)
    try:
        cursor = check_conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        status_row = cursor.fetchone()
        if not status_row or status_row[0] != "ok":
            raise RuntimeError(f"Source backup is corrupted or invalid: {status_row}")
        logger.info("[✓] Backup file integrity verified.")
    finally:
        check_conn.close()

    # Step 2: Create a safety pre-restore snapshot of current database if present
    if os.path.exists(target_db_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pre_restore_snapshot = f"{target_db_path}.pre_restore_{timestamp}.bak"
        logger.info(f"Step 2: Creating safety snapshot of current database: '{pre_restore_snapshot}'...")
        shutil.copy2(target_db_path, pre_restore_snapshot)

    # Step 3: Copy backup file to target location
    logger.info(f"Step 3: Restoring database to '{target_db_path}'...")
    os.makedirs(os.path.dirname(target_db_path) or ".", exist_ok=True)
    shutil.copy2(source_backup_path, target_db_path)

    # Remove stale WAL / SHM files if any
    for ext in ["-wal", "-shm"]:
        sidecar = f"{target_db_path}{ext}"
        if os.path.exists(sidecar):
            try:
                os.remove(sidecar)
            except OSError:
                pass

    # Step 4: Final verification on restored database
    logger.info("Step 4: Running post-restore integrity check...")
    restored_conn = sqlite3.connect(target_db_path)
    try:
        cursor = restored_conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        status_row = cursor.fetchone()
        if not status_row or status_row[0] != "ok":
            raise RuntimeError(f"Restored database failed integrity check: {status_row}")
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tbl_count = cursor.fetchone()[0]
        logger.info(f"[SUCCESS] Restore completed successfully! ({tbl_count} tables found in restored database)")
    finally:
        restored_conn.close()


def main():
    parser = argparse.ArgumentParser(description="TabSynth AI SQLite Database Restore Utility")
    parser.add_argument("--source", required=True, help="Path to verified backup archive file (.db)")
    parser.add_argument("--target", default=settings.db_path, help="Path to destination database")
    args = parser.parse_args()

    try:
        restore_database(source_backup_path=args.source, target_db_path=args.target)
    except Exception as exc:
        logger.error(f"[ERROR] Database restore failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
