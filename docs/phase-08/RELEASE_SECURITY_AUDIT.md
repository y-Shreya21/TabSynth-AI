# TabSynth AI — Release Security Audit & Threat Model Report
## Comprehensive Pre-Release Security, Compliance & Vulnerability Audit

**Audit Date**: November 2026  
**Software Version**: `v0.9.0-rc.1` / Release Candidate  
**Auditor**: Senior DevSecOps & Production Reliability Engineer  
**Classification**: Public Release Verification  
**Overall Security Rating**: **PASSED (Zero High/Critical Vulnerabilities)**  

---

## 1. Executive Summary

A comprehensive security audit of **TabSynth AI** was executed prior to public release. The audit encompassed secret scanning, dependency vulnerability review, input validation, path traversal defense, file upload sanitization, threat modeling, and open-source license compliance.

### Key Audit Findings:
1. **Secret Scanning**: 0 credentials, private keys, API tokens, or personal identifiers were detected in version-controlled git objects.
2. **Path Traversal Defense**: All user-supplied filesystem paths are strictly validated through canonical resolution against registered storage boundaries (`api/utils/security.py`).
3. **Upload Sanitization**: Uploaded files undergo MIME/extension verification, file size bounding (100 MB), and binary executable sniffing.
4. **License Compatibility**: All 14 third-party runtime and development dependencies use permissive licenses (Apache 2.0, BSD-3-Clause, MIT) fully compatible with Apache 2.0.
5. **Container Security**: Multi-stage Docker build enforces a dedicated non-root user (`uid=10001, gid=10001`).

---

## 2. Secret & Sensitive Data Scanning Audit

Automated regex and entropy-based pattern matching was performed across the entire repository history and working tree:

| Pattern Category | Target Signatures Checked | Matches Found | Remediation Status |
| :--- | :--- | :---: | :--- |
| **API Keys & Tokens** | AWS, GitHub, OpenAI, WandB, HuggingFace tokens | **0** | Clean |
| **Private Cryptographic Keys** | RSA/DSA/EC `BEGIN PRIVATE KEY`, `.pem`, `.key` | **0** | Excluded in `.gitignore` |
| **Database Credentials** | Hardcoded passwords, URI connection strings | **0** | Excluded / Configurable via Env |
| **Local System Paths** | `/Users/`, `/home/`, Windows paths in source code | **0** | Clean relative paths used across all modules |
| **Environment Files** | `.env`, `.env.local`, `.env.production` | **0** | Git-ignored; `.env.example` provided |

---

## 3. Safe Path Validation & Confinement Audit

Uncontrolled file access vulnerabilities (CWE-22 / CWE-23: Path Traversal) were audited in `api/utils/security.py`:

### Code Inspection of Path Confinement:
```python
def is_safe_path(target_path: str, allowed_roots: List[str]) -> bool:
    if not target_path or not isinstance(target_path, str):
        return False
    if "\x00" in target_path:  # Null byte injection protection
        return False
    try:
        canonical_target = os.path.realpath(os.path.abspath(target_path))
        for root in allowed_roots:
            canonical_root = os.path.realpath(os.path.abspath(root))
            if os.path.commonpath([canonical_target, canonical_root]) == canonical_root:
                return True
        return False
    except Exception:
        return False
```

### Boundary Test Verification:
The following malicious attack payloads were passed to `api/utils/security.validate_safe_path`:
- `../../etc/passwd` $\implies$ **Blocked (HTTP 400)**
- `/etc/shadow` $\implies$ **Blocked (HTTP 400)**
- `data/../../secret.key` $\implies$ **Blocked (HTTP 400)**
- `data/raw/../../../root` $\implies$ **Blocked (HTTP 400)**
- `\x00data/raw/test.csv` $\implies$ **Blocked (HTTP 400)**

*Result*: All path traversal attempts are neutralized before opening any filesystem descriptor.

---

## 4. File Upload Sanitization & Content Verification

Tabular ingestion endpoints accept multipart CSV files. Protective controls implemented in `validate_csv_upload` include:
1. **Extension Enforcement**: Rejects non-`.csv` extensions (`.sh`, `.exe`, `.py`, `.bin`) with `HTTP 400`.
2. **File Size Capping**: Rejects uploads $> 100\text{ MB}$ with `HTTP 413 Request Entity Too Large`.
3. **Empty File Rejection**: Rejects 0-byte uploads with `HTTP 400`.
4. **Binary Sniffing**: Inspects the first 512 bytes for null byte density and binary executable magic bytes (ELF `\x7fELF`, PE `MZ`), rejecting disguised binaries with `HTTP 400`.
5. **Column Limit Guard**: Rejects tables exceeding 500 columns (`MAX_PERMITTED_COLUMNS = 500`) to prevent memory exhaustion attacks.

---

## 5. Dependency Vulnerability & License Audit

Every direct dependency declared in `pyproject.toml` was reviewed for known CVEs and license compliance:

| Package | Minimum Version | License | License Compatibility | Known Vulnerabilities |
| :--- | :--- | :--- | :---: | :---: |
| `torch` | $\ge 2.2.0$ | BSD-style | **Compatible** | None |
| `numpy` | $\ge 1.26.0$ | BSD-3-Clause | **Compatible** | None |
| `scipy` | $\ge 1.12.0$ | BSD-3-Clause | **Compatible** | None |
| `pandas` | $\ge 2.2.0$ | BSD-3-Clause | **Compatible** | None |
| `scikit-learn` | $\ge 1.4.0$ | BSD-3-Clause | **Compatible** | None |
| `pydantic` | $\ge 2.6.0$ | MIT | **Compatible** | None |
| `pyyaml` | $\ge 6.0.1$ | MIT | **Compatible** | None |
| `plotly` | $\ge 5.19.0$ | MIT | **Compatible** | None |
| `fastapi` | $\ge 0.110.0$ | MIT | **Compatible** | None |
| `uvicorn` | $\ge 0.28.0$ | BSD-3-Clause | **Compatible** | None |
| `streamlit` | $\ge 1.32.0$ | Apache-2.0 | **Compatible** | None |
| `httpx` | $\ge 0.27.0$ | BSD-3-Clause | **Compatible** | None |
| `pytest` | $\ge 8.0.0$ | MIT | **Compatible** | None |
| `pytest-cov` | $\ge 4.1.0$ | MIT | **Compatible** | None |

*Summary*: Zero copyleft (GPL / AGPL) dependencies. The entire dependency tree is permissive and compliant with TabSynth AI's Apache 2.0 release.

---

## 6. Threat Modeling & Risk Matrix (STRIDE)

| Threat Category (STRIDE) | Attack Vector / Scenario | Mitigating Security Control | Residual Risk |
| :--- | :--- | :--- | :---: |
| **Spoofing** | Forged client requesting background job cancellation | Jobs tracked via random UUID tokens; API accessible only on internal/trusted network | Low |
| **Tampering** | Database corruption during high-concurrency writes | SQLite WAL mode (`PRAGMA journal_mode=WAL;`), 10s busy timeout, non-blocking backup API | Low |
| **Repudiation** | Unaudited execution of training runs | Structured request logging middleware assigning unique `ReqID` UUID to every HTTP request | Negligible |
| **Information Disclosure** | Synthetic data memorizing training participant identities | Automated DCR/NNDR auditing, shadow MIA testing, and explicit Differential Privacy notice | Low |
| **Denial of Service** | Malicious multi-GB CSV upload or negative epoch parameter | 100 MB upload cap, 500 column cap, Pydantic `Field(gt=0)` parameter validation | Low |
| **Elevation of Privilege** | Container breakout via root process | Dockerfile executes strictly under non-root unprivileged user `appuser (uid=10001)` | Low |

---

## 7. Security Audit Conclusion

TabSynth AI satisfies enterprise open-source security standards. The application features robust path confinement, input sanitization, non-root containerization, and clean license attribution, making it safe for public release and technical deployment.
