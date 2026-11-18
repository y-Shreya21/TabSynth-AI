# TabSynth Analytics: Security Policy & Threat Model

**Project:** TabSynth AI / TabSynth Analytics  
**Version:** 1.0.0  
**Status:** Production Hardened  

---

## 1. Security Architecture & Principles

TabSynth Analytics is engineered according to the principle of least privilege, defense-in-depth, and strict input confinement:

1. **Path Confinement**: All client-specified paths must be validated and confined to approved application directories. Path traversal via `../`, symlinks, or absolute system paths is rejected.
2. **File Ingestion Guards**: File uploads are restricted by size (default: 50MB), mandatory `.csv` extensions, MIME type sniff checks, and schema sanity rules (max 500 columns, duplicate column detection).
3. **No Raw Content in Logs**: Raw uploaded tabular rows and sensitive data fields are never written to server execution logs or audit traces.
4. **Exception Masking**: Server exceptions returned to API clients are sanitized to generic error messages referencing correlation Request IDs (`[ReqID: ...]`). Internal filesystem paths and stack traces are suppressed in HTTP responses.
5. **Safe Deserialization**: Checkpoint loading strictly verifies internal directory provenance and whitelists authorized model components. External, unverified `.pt` or pickle files are rejected.
6. **Least Privilege Containerization**: Docker containers execute as a dedicated non-root user (`tabsynth`, UID 1000).

---

## 2. Threat Model & Mitigations

| Threat | Risk Level | Mitigation Strategy | Implementation |
| :--- | :--- | :--- | :--- |
| **Path Traversal / Arbitrary File Read** | Critical | Verify that all resolved file paths share a canonical directory with approved roots (`data/`, `checkpoints/`, `experiments/`). | `api/utils/security.py:validate_safe_path()` |
| **Denial of Service via Huge Upload** | High | Enforce pre-read file size limits and chunk-based size validation before allocating memory. | `DatasetService.upload_dataset()`, `MAX_UPLOAD_SIZE_BYTES` |
| **CSV Bomb / Memory Exhaustion** | High | Validate column count (< 500), header lengths, and parse using bounded memory buffers. | `DatasetService.upload_dataset()` |
| **Arbitrary Code Execution via Deserialization** | Critical | Confine PyTorch checkpoints to authorized internal paths; register safe globals with PyTorch 2.4+. | `CheckpointManager.load()`, `is_safe_path()` |
| **Cross-Origin Request Forgery / Leakage** | Medium | Configurable CORS origins with environment parsing; strict disallowal of `*` in production mode. | `api/config.py`, `CORSMiddleware` |
| **Information Disclosure via Stack Traces** | Medium | Centralized exception handling intercepting unhandled errors and masking internal details. | `api/middleware/error_handler.py` |
| **Privilege Escalation inside Container** | High | Drop root privileges during container build; run as non-root user `tabsynth`. | `Dockerfile` |

---

## 3. Safe Path Confinement Specification

The security module `api/utils/security.py` enforces canonical path containment:

```python
from api.utils.security import validate_safe_path

# Raises HTTPException(400) if path attempts to escape allowed storage roots
safe_real_path = validate_safe_path(
    path=request.real_data_path,
    allowed_roots=[settings.data_dir, settings.raw_data_dir, settings.synthetic_data_dir],
)
```

Rules enforced:
- Resolves all symlinks via `os.path.realpath()`.
- Prohibits null byte injection (`\x00`).
- Validates that `os.path.commonpath([resolved_path, allowed_dir]) == allowed_dir`.
- Rejects paths containing relative traversal tokens (`..`).

---

## 4. Reporting Security Vulnerabilities

If you identify a security vulnerability in TabSynth AI, please disclose it responsibly:
- **Email**: `security@tabsynth.ai` (or via private GitHub Vulnerability Reporting)
- Please do not file public GitHub issues for security vulnerabilities.
- Provide a proof-of-concept and environment reproduction details.
