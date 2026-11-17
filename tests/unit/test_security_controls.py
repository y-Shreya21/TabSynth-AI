"""Unit tests for security controls, path traversal prevention, and upload sanitization."""

import io
import pytest
from fastapi import HTTPException
from api.services.dataset_service import DatasetService
from api.config import settings


def test_sanitize_filename():
    assert DatasetService._sanitize_filename("../../etc/passwd") == "passwd"
    assert DatasetService._sanitize_filename("..\\..\\windows\\system32\\calc.exe") == "calc.exe"
    assert DatasetService._sanitize_filename("my data (1) [final].csv") == "my_data__1___final_.csv"
    assert DatasetService._sanitize_filename("") == "dataset.csv"


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
