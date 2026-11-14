# Contributing to TabSynth AI

Thank you for your interest in contributing to **TabSynth AI**! As a research-oriented generative deep learning project, we maintain high standards of code quality, mathematical correctness, reproducibility, and data privacy.

---

## 1. Code of Conduct

We are dedicated to providing a welcoming, inclusive, and harassment-free environment for everyone. Please be respectful and constructive in all issues, pull requests, and discussions.

---

## 2. Development Setup

1. **Fork and clone** the repository:
   ```bash
   git clone https://github.com/y-Shreya21/TabSynth-AI.git
   cd TabSynth-AI
   ```
2. **Create a virtual environment** using Python 3.11:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install the package in editable mode** with development dependencies:
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev,serving]"
   ```
4. **Verify the installation** by running the test suite:
   ```bash
   pytest tests/
   python scripts/run_phase05_smoke.py
   ```

---

## 3. Contribution Workflow

1. Create a descriptive topic branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Ensure strict adherence to project standards:
   - **Zero Data Leakage:** Ensure preprocessors and scalers only ever fit on training data splits.
   - **No Hardcoded Absolute Paths:** Use project-relative paths.
   - **Type Hints & Docstrings:** Provide full type annotations and Google/Sphinx style docstrings.
   - **Reproducibility:** Seed all stochastic operations (`torch.manual_seed`, `np.random.seed`).
3. Add unit or integration tests for any new features or bug fixes under `tests/`.
4. Ensure all 103 existing tests continue to pass:
   ```bash
   pytest tests/ -v
   ```
5. Follow Conventional Commits:
   - `feat(...)`: New feature or capability.
   - `fix(...)`: Bug fix.
   - `docs(...)`: Documentation additions or revisions.
   - `test(...)`: Adding or updating test cases.
   - `chore(...)`: Tooling, configuration, or packaging maintenance.
6. Push your branch and open a Pull Request.

---

## 4. Privacy & Security Rules

- **NEVER commit raw or private datasets** to Git.
- **NEVER commit sensitive generated records** or patient/customer microdata.
- **NEVER commit API tokens, passwords, or credentials.**
- Respect the formal privacy boundaries documented in `SECURITY.md` and `docs/PHASE_05_EVALUATION_REPORT.md`.
