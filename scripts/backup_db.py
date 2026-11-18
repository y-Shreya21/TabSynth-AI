#!/usr/bin/env python3
"""Online zero-downtime SQLite backup utility with integrity check and retention rotation."""

import os
import sys
import glob
import sqlite3
import argparse
import datetime
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.config import settings

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("tabsynth.backup")


def create_backup(
    db_path: str = settings.db_path,
    dest_dir: str = settings.backup_dir,
    retention: int = 7,
) -> str:
    """Creates a verified timestamped backup of the SQLite database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Source database not found at: {db_path}")

    os.makedirs(dest_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_filename = f"tabsynth_backup_{timestamp}.db"
    backup_path = os.path.join(dest_dir, backup_filename)

    logger.info(f"Initiating online backup from '{db_path}' to '{backup_path}'...")

    # Step 1: Perform online backup
    src_conn = sqlite3.connect(db_path, timeout=30.0)
    dest_conn = sqlite3.connect(backup_path)
    try:
        with dest_conn:
            src_conn.backup(dest_conn, pages=100)
    finally:
        src_conn.close()
        dest_conn.close()

    # Step 2: Run SQLite integrity check on the backup file
    logger.info("Running integrity check on created backup file...")
    check_conn = sqlite3.connect(backup_path)
    try:
        cursor = check_conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        status_row = cursor.fetchone()
        if not status_row or status_row[0] != "ok":
            raise RuntimeError(f"Backup integrity check failed: {status_row}")
        logger.info("[✓] Integrity check PASSED: database is healthy.")
    finally:
        check_conn.close()

    # Step 3: Rotate and prune older backups exceeding retention count
    existing_backups = sorted(glob.glob(os.path.join(dest_dir, "tabsynth_backup_*.db")))
    if len(existing_backups) > retention:
        to_prune = existing_backups[:-retention]
        for old_backup in to_prune:
            try:
                os.remove(old_backup)
                logger.info(f"Pruned historical backup: '{os.path.basename(old_backup)}'")
            except OSError as err:
                logger.warning(f"Could not remove old backup '{old_backup}': {err}")

    backup_size_kb = os.path.getsize(backup_path) / 1024.0
    logger.info(f"[SUCCESS] Backup created successfully: '{backup_path}' ({backup_size_kb:.2f} KB)")
    return backup_path


def main():
    parser = argparse.ArgumentParser(description="TabSynth AI SQLite Online Backup Utility")
    parser.add_argument("--db-path", default=settings.db_path, help="Path to source database file")
    parser.add_argument("--dest-dir", default=settings.backup_dir, help="Directory to store backups")
    parser.add_argument("--retention", type=int, default=7, help="Number of recent backups to retain")
    args = parser.parse_args()

    try:
        create_backup(db_path=args.db_path, dest_dir=args.dest_dir, retention=args.retention)
    except Exception as exc:
        logger.error(f"[ERROR] Database backup failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
