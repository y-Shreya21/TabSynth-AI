"""Security utilities for input validation, path confinement, and upload protection."""

import os
import re
import logging
from typing import List, Optional
from fastapi import HTTPException, status

logger = logging.getLogger("tabsynth.security")

# Maximum permitted columns in uploaded tabular dataset to avoid memory bombs
MAX_PERMITTED_COLUMNS = 500


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and special character injection."""
    if not filename:
        return "dataset.csv"
    # Remove path elements
    base = os.path.basename(filename.replace("\\", "/"))
    # Allow alphanumeric, underscore, hyphen, and period
    sanitized = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", base)
    if not sanitized.lower().endswith(".csv"):
        sanitized = f"{sanitized}.csv"
    return sanitized or "dataset.csv"


def is_safe_path(target_path: str, allowed_roots: List[str]) -> bool:
    """Check if target_path resolves strictly within one of the allowed_roots without traversal."""
    if not target_path or not isinstance(target_path, str):
        return False

    # Block null byte injection
    if "\x00" in target_path:
        return False

    try:
        canonical_target = os.path.realpath(os.path.abspath(target_path))
        for root in allowed_roots:
            if not root:
                continue
            canonical_root = os.path.realpath(os.path.abspath(root))
            # Target must either equal the root or be a child directory/file of root
            try:
                common = os.path.commonpath([canonical_target, canonical_root])
                if common == canonical_root:
                    return True
            except ValueError:
                # Different drives on Windows or incompatible paths
                continue
        return False
    except Exception as exc:
        logger.warning(f"Error checking path safety for '{target_path}': {exc}")
        return False


def validate_safe_path(
    target_path: str,
    allowed_roots: List[str],
    error_message: Optional[str] = None,
) -> str:
    """Validate that target_path is within allowed_roots; raises HTTPException(400) if unsafe."""
    if not is_safe_path(target_path, allowed_roots):
        detail_msg = error_message or f"Access denied: Path '{target_path}' is outside permitted directories."
        logger.warning(f"Path traversal attempt blocked: '{target_path}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail_msg,
        )
    return os.path.realpath(os.path.abspath(target_path))


def validate_csv_upload(
    filename: str,
    file_bytes: bytes,
    max_size_bytes: int = 50 * 1024 * 1024,
) -> None:
    """Validate an uploaded CSV file against size, extension, and content rules."""
    # 1. Filename extension check
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Only tabular CSV (.csv) files are permitted. Received: '{filename}'",
        )

    # 2. File size bounds
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(file_bytes) > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds the maximum permitted limit of {max_mb:.1f} MB.",
        )

    # 3. Binary executable check (sniff for ELF, PE headers, or excessive null bytes)
    sample = file_bytes[:1024]
    if b"\x00" in sample:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file appears to be a binary file, not a valid tabular CSV.",
        )
