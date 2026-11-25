"""Unit tests for the automated security and data privacy scanner."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.security_audit import (
    audit_privacy,
    audit_secrets,
    scan_file_for_secrets,
    should_scan_file,
    WHITELISTED_CHECKPOINTS,
)


def test_should_scan_file():
    # Included source and configuration files
    assert should_scan_file("src/tabsynth/model.py")
    assert should_scan_file("api/server.py")
    assert should_scan_file("app/main.py")
    assert should_scan_file("configs/model.yaml")
    assert should_scan_file("Dockerfile")
    assert should_scan_file(".env.example")

    # Excluded directories and file patterns
    assert not should_scan_file(".git/config")
    assert not should_scan_file(".venv/lib/python3.11/site-packages/pkg.py")
    assert not should_scan_file("artifacts/model.pt")
    assert not should_scan_file("docs/phase-08/RELEASE_SECURITY_AUDIT.md")
    assert not should_scan_file("README.md")
    assert not should_scan_file("scripts/security_audit.py")
    assert not should_scan_file("tests/unit/test_security_audit.py")


def test_scan_file_for_secrets_detection():
    # Test fixture with intentional test tokens
    with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as f:
        f.write("# Dummy test code\n")
        f.write("def harmless():\n")
        f.write("    return True\n")
        f.write("fake_token = 'api_key = \"abcedf12345678901234567890\"'\n")
        f.write("pem_header = '-----BEGIN RSA PRIVATE KEY-----'\n")
        f.write("aws_id = 'AKIAIOSFODNN7EXAMPLE'\n")
        f.write("gh_token = 'ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\n")
        temp_path = Path(f.name)

    try:
        violations = scan_file_for_secrets(temp_path)
        assert len(violations) == 4
        patterns = [v[1] for v in violations]
        assert "Generic High-Entropy API Key / Token" in patterns
        assert "PEM Private Key Header" in patterns
        assert "AWS Access Key ID" in patterns
        assert "GitHub Personal Access Token" in patterns
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_scan_file_for_secrets_clean():
    with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as f:
        f.write("# Clean file\n")
        f.write("VALUE = 42\n")
        f.write("def compute():\n")
        f.write("    return VALUE * 2\n")
        temp_path = Path(f.name)

    try:
        violations = scan_file_for_secrets(temp_path)
        assert len(violations) == 0
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_audit_privacy_whitelisted_and_violations():
    # Test that whitelisted checkpoints pass
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "\n".join([
            "checkpoints/adult_hybrid_best.pt",
            "checkpoints/adult_mlp_best.pt",
            "checkpoints/.gitkeep",
            "src/tabsynth/model.py",
        ])
        assert audit_privacy(Path("."))

    # Test that raw CSV triggers privacy violation
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "\n".join([
            "data/raw/confidential_records.csv",
            "src/tabsynth/model.py",
        ])
        assert not audit_privacy(Path("."))

    # Test that synthetic CSV triggers privacy violation
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "\n".join([
            "data/synthetic/generated_patients.csv",
            "src/tabsynth/model.py",
        ])
        assert not audit_privacy(Path("."))

    # Test that unauthorized checkpoint triggers violation
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "\n".join([
            "checkpoints/unauthorized_model_epoch_99.pt",
            "src/tabsynth/model.py",
        ])
        assert not audit_privacy(Path("."))


def test_current_repo_passes_audit():
    repo_root = Path(__file__).resolve().parent.parent.parent
    assert audit_secrets(repo_root)
    assert audit_privacy(repo_root)
