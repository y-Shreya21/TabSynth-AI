# Security & Data Privacy Policy

## 1. Scope & Research Purpose

**TabSynth AI** is an advanced open-source machine learning research framework designed for synthetic tabular data generation. Protecting sensitive data, preventing credential leakage, and upholding rigorous standards of privacy auditing are central to the project's mission.

---

## 2. Reporting a Vulnerability

If you discover a potential security vulnerability, credential disclosure, or private data exposure within this repository, please **do not open a public issue**.

Instead, please responsibly disclose the vulnerability by contacting the maintainers directly or emailing:
- **Email:** `security@tabsynth-ai.org` (or contact repository owners via GitHub Security Advisories)

We aim to acknowledge receipt of security reports within 48 hours and provide a fix or remediation plan within 7 business days.

---

## 3. Data Privacy Boundaries & Formal Non-DP Notice

> **FORMAL PRIVACY NOTICE**:  
> Empirical privacy metrics implemented in TabSynth AI (Distance-to-Closest-Record, Nearest Neighbor Distance Ratios, Membership Inference Attack ROC-AUC, Attribute Inference Advantage, and Singling-Out Risk) evaluate empirical memorization and reconstruction vulnerability on specific finite sample partitions.  
> These empirical diagnostics **DO NOT** provide provable, worst-case mathematical Differential Privacy guarantees $(\epsilon, \delta)$. Provable mathematical guarantees require formal differential privacy mechanisms (such as DP-SGD with calibrated Gaussian noise addition and per-sample gradient clipping) during model training.

---

## 4. Repository Security Guidelines

1. **No Real Sensitive Data in Commits:**
   - Raw datasets (`data/raw/*.csv`), generated records (`data/synthetic/*.csv`), and customer/patient microdata are strictly excluded by `.gitignore`.
2. **No Credentials in Commits:**
   - Secrets, private keys (`*.pem`, `*.key`), and environment variables (`.env`, `.env.*`) are strictly excluded.
3. **Automated Secret & Data Scanning:**
   - Continuous Integration workflows continuously verify that no CSV data files or secret token patterns are tracked in Git.
