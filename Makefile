.PHONY: help install test smoke benchmark lint clean run-api run-app

PYTHON ?= python3

help:
	@echo "TabSynth AI - Developer Workflow Commands:"
	@echo "  make install     - Install package in editable mode with development dependencies"
	@echo "  make test        - Run complete pytest test suite (103 tests)"
	@echo "  make smoke       - Run fast evaluation smoke test (< 2s)"
	@echo "  make benchmark   - Run multi-model benchmark evaluation on Adult Income"
	@echo "  make ablations   - Run research ablation experiments (E1, E5, E6, E7)"
	@echo "  make report      - Generate docs/PHASE_05_EVALUATION_REPORT.md"
	@echo "  make run-api     - Launch FastAPI REST service on port 8000"
	@echo "  make run-app     - Launch Streamlit interactive dashboard on port 8501"
	@echo "  make clean       - Remove cached bytecode and test artifacts"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

smoke:
	$(PYTHON) scripts/run_phase05_smoke.py

benchmark:
	$(PYTHON) scripts/run_phase05_benchmark.py --num_samples 500

ablations:
	$(PYTHON) scripts/run_phase05_ablations.py --num_samples 500

report:
	$(PYTHON) scripts/generate_phase05_report.py

run-api:
	uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

run-app:
	streamlit run app/main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/ build/ dist/ *.egg-info
