"""Unit tests for security controls, path traversal prevention, and upload sanitization."""

import os
import pytest
from fastapi import HTTPException
from api.services.dataset_service import DatasetService
from api.config import settings
from api.utils.security import (
    is_safe_path,
    validate_safe_path,
    sanitize_filename,
    validate_csv_upload,
)


def test_sanitize_filename():
    assert DatasetService._sanitize_filename("../../etc/passwd") == "passwd"
    assert DatasetService._sanitize_filename("..\\..\\windows\\system32\\calc.exe") == "calc.exe"
    assert DatasetService._sanitize_filename("my data (1) [final].csv") == "my_data__1___final_.csv"
    assert DatasetService._sanitize_filename("") == "dataset.csv"

    # api.utils.security sanitize_filename
    assert sanitize_filename("../../etc/passwd").endswith(".csv")
    assert ".." not in sanitize_filename("../../test.csv")
    assert sanitize_filename("safe_name.csv") == "safe_name.csv"


def test_path_traversal_guards():
    allowed_roots = [settings.data_dir, settings.checkpoint_dir]

    # Blocked paths
    assert not is_safe_path("../../etc/passwd", allowed_roots)
    assert not is_safe_path("/etc/shadow", allowed_roots)
    assert not is_safe_path("data/../../secret.txt", allowed_roots)
    assert not is_safe_path("\x00data/raw/file.csv", allowed_roots)
    assert not is_safe_path("", allowed_roots)

    with pytest.raises(HTTPException) as exc:
        validate_safe_path("../../etc/passwd", allowed_roots)
    assert exc.value.status_code == 400

    # Allowed paths
    safe_path = os.path.join(settings.data_dir, "raw", "dataset.csv")
    assert is_safe_path(safe_path, allowed_roots)
    validated = validate_safe_path(safe_path, allowed_roots)
    assert os.path.isabs(validated)


def test_csv_upload_validation():
    # Extension validation
    with pytest.raises(HTTPException) as exc:
        validate_csv_upload("malicious.sh", b"echo hello")
    assert exc.value.status_code == 400

    # Empty validation
    with pytest.raises(HTTPException) as exc:
        validate_csv_upload("empty.csv", b"")
    assert exc.value.status_code == 400

    # Binary sniffing
    with pytest.raises(HTTPException) as exc:
        validate_csv_upload("binary.csv", b"\x00\x01\x02\x03fakebinary")
    assert exc.value.status_code == 400

    # Size limit
    with pytest.raises(HTTPException) as exc:
        validate_csv_upload("huge.csv", b"a" * 100, max_size_bytes=50)
    assert exc.value.status_code == 413

    # Valid CSV
    validate_csv_upload("good.csv", b"a,b\n1,2\n")


def test_upload_size_limit():
    huge_data = b"a" * (settings.max_upload_size_bytes + 1024)
    with pytest.raises(HTTPException) as exc:
        DatasetService.upload_dataset(huge_data, "huge.csv")
    assert exc.value.status_code == 413


def test_empty_file_rejected():
    with pytest.raises(HTTPException) as exc:
        DatasetService.upload_dataset(b"", "empty.csv")
    assert exc.value.status_code == 400


def test_duplicate_columns_rejected():
    csv_dup = b"col1,col1,col2\n1,2,3\n4,5,6\n"
    with pytest.raises(HTTPException) as exc:
        DatasetService.upload_dataset(csv_dup, "dup.csv")
    assert exc.value.status_code == 400
    assert "duplicate" in exc.value.detail.lower()


def test_invalid_target_column_rejected():
    csv_data = b"col1,col2\n1,2\n3,4\n"
    with pytest.raises(HTTPException) as exc:
        DatasetService.upload_dataset(csv_data, "data.csv", target_column="nonexistent_target")
    assert exc.value.status_code == 400
    assert "does not exist" in exc.value.detail.lower()
