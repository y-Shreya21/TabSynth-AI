#!/usr/bin/env python3
"""
TabSynth AI - Automated Security & Data Privacy Audit Scanner

This utility audits the codebase and repository tracking for:
1. High-risk secret patterns (PEM private keys, API keys, tokens, credentials).
2. Data privacy compliance (ensuring raw/synthetic CSV datasets and unauthorized
   checkpoints are not tracked in git).

Usage:
    python scripts/security_audit.py                 # Run both secret & privacy checks
    python scripts/security_audit.py --check-secrets  # Run secret scanning only
    python scripts/security_audit.py --check-privacy  # Run git privacy inspection only
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Directories and file patterns excluded from secret scanning
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "artifacts",
    "docs",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".agents",
    ".claude",
    ".idea",
    ".vscode",
}

EXCLUDED_FILES = {
    "security_audit.py",
    "test_security_audit.py",
}

EXCLUDED_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".pdf",
    ".docx",
    ".html",
    ".pyc",
    ".pyo",
    ".pyd",
    ".pt",
    ".pth",
    ".sqlite",
    ".db",
}

# Regex signatures for secret detection (compiled)
SECRET_PATTERNS = [
    (
        "PEM Private Key Header",
        re.compile(r"-----BEGIN (?:[A-Z0-9_-]+ )?PRIVATE KEY-----"),
    ),
    (
        "RSA Private Key Signature",
        re.compile(r"-----BEGIN RSA PRIVATE KEY-----"),
    ),
    (
        "Generic High-Entropy API Key / Token",
        re.compile(
            r"""(?i)(?:api[_-]?key|secret[_-]?key|auth[_-]?token|access[_-]?token|private[_-]?key)\s*[:=]\s*['"][a-zA-Z0-9_\-\.]{20,}['"]"""
        ),
    ),
    (
        "AWS Access Key ID",
        re.compile(r"""\b(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b"""),
    ),
    (
        "GitHub Personal Access Token",
        re.compile(r"""\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}\b"""),
    ),
    (
        "Slack Token Signature",
        re.compile(r"""\bxox[baprs]-[0-9]{12}-[0-9]{12}-[a-zA-Z0-9]{24}\b"""),
    ),
]

# Legitimate benchmark checkpoints whitelisted for git tracking
WHITELISTED_CHECKPOINTS = {
    "checkpoints/adult_hybrid_best.pt",
    "checkpoints/adult_mlp_best.pt",
    "checkpoints/.gitkeep",
}


def should_scan_file(rel_path: str) -> bool:
    """Determine whether a file should be scanned for credentials."""
    p = Path(rel_path)
    parts = p.parts
    for excluded in EXCLUDED_DIRS:
        if excluded in parts:
            return False

    if p.name in EXCLUDED_FILES:
        return False

    ext = p.suffix.lower()
    if ext in EXCLUDED_EXTENSIONS:
        return False

    return True


def scan_file_for_secrets(file_path: Path) -> List[Tuple[int, str, str]]:
    """Scan a single file for secret patterns. Returns (line_no, pattern_name, line_snippet)."""
    violations = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                for pattern_name, regex in SECRET_PATTERNS:
                    if regex.search(line):
                        redacted = stripped[:80] + ("..." if len(stripped) > 80 else "")
                        violations.append((idx, pattern_name, redacted))
                        break  # Report at most one violation per line
    except Exception as e:
        print(f"  [!] Warning: Failed to read {file_path}: {e}", file=sys.stderr)
    return violations


def audit_secrets(root_dir: Path) -> bool:
    """Scan the repository files for potential secret leaks."""
    print("[*] Starting Secret & Credential Scanning Audit...")
    total_scanned = 0
    all_violations = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Prune excluded directories in-place for performance
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]

        for filename in filenames:
            abs_path = Path(dirpath) / filename
            try:
                rel_path = abs_path.relative_to(root_dir).as_posix()
            except ValueError:
                rel_path = str(abs_path)

            if not should_scan_file(rel_path):
                continue

            total_scanned += 1
            file_violations = scan_file_for_secrets(abs_path)
            for line_no, pattern_name, snippet in file_violations:
                all_violations.append((rel_path, line_no, pattern_name, snippet))

    print(f"[*] Scanned {total_scanned} candidate source and configuration files.")

    if all_violations:
        print(f"\n[!] SECURITY VIOLATION: Found {len(all_violations)} potential secret pattern(s):", file=sys.stderr)
        for rel_path, line_no, pattern_name, snippet in all_violations:
            print(f"    - {rel_path}:{line_no} [{pattern_name}]: {snippet}", file=sys.stderr)
        return False

    print("[+] PASS: No credential or private key patterns detected.")
    return True


def audit_privacy(root_dir: Path) -> bool:
    """Inspect git tracked files to ensure no sensitive raw/synthetic CSV data or unauthorized checkpoints are committed."""
    print("[*] Starting Git Data Privacy & Checkpoint Inspection...")
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=str(root_dir),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as e:
        print(f"[!] Error running git ls-files: {e}", file=sys.stderr)
        return False

    tracked_files = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]

    privacy_violations = []

    raw_csv_pattern = re.compile(r"^data/raw/.*\.csv$")
    synthetic_csv_pattern = re.compile(r"^data/synthetic/.*\.csv$")
    checkpoint_pattern = re.compile(r"^checkpoints/.*\.pt$")

    for file_path in tracked_files:
        norm_path = file_path.replace("\\", "/")

        # Check raw CSV data files
        if raw_csv_pattern.match(norm_path):
            privacy_violations.append((norm_path, "Raw Training Dataset (Privacy Leak)"))

        # Check synthetic CSV data files
        elif synthetic_csv_pattern.match(norm_path):
            privacy_violations.append((norm_path, "Synthetic Dataset (Data Privacy Violation)"))

        # Check checkpoint models
        elif checkpoint_pattern.match(norm_path):
            if norm_path not in WHITELISTED_CHECKPOINTS:
                privacy_violations.append((norm_path, "Unauthorized Model Checkpoint (Artifact Violation)"))

    if privacy_violations:
        print(f"\n[!] DATA PRIVACY VIOLATION: Found {len(privacy_violations)} unauthorized tracked file(s):", file=sys.stderr)
        for path, reason in privacy_violations:
            print(f"    - {path} [{reason}]", file=sys.stderr)
        return False

    print("[+] PASS: No raw/synthetic CSV data or unauthorized checkpoints are tracked in Git.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="TabSynth AI Security & Privacy Audit Scanner")
    parser.add_argument("--check-secrets", action="store_true", help="Run secret scanning only")
    parser.add_argument("--check-privacy", action="store_true", help="Run git privacy inspection only")
    parser.add_argument("--root-dir", type=str, default=".", help="Root repository path")
    args = parser.parse_args()

    root_path = Path(args.root_dir).resolve()

    run_all = not args.check_secrets and not args.check_privacy
    success = True

    if run_all or args.check_secrets:
        if not audit_secrets(root_path):
            success = False

    if run_all or args.check_privacy:
        if not audit_privacy(root_path):
            success = False

    if not success:
        print("\n[FAILED] Security/Privacy audit failed.", file=sys.stderr)
        return 1

    print("\n[SUCCESS] All security and privacy audit checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
