.PHONY: help install test smoke smoke-phase06 smoke-phase07 benchmark ablations report run-api run-app docker-build docker-up docker-down backup restore check-config clean

PYTHON ?= python3

help:
	@echo "TabSynth AI - Developer & DevOps Workflow Commands:"
	@echo "  make install         - Install package in editable mode with full dependencies"
	@echo "  make test            - Run complete pytest test suite (120+ tests)"
	@echo "  make smoke           - Run fast Phase 05 evaluation smoke test (< 2s)"
	@echo "  make smoke-phase06   - Run Phase 06 application end-to-end smoke test"
	@echo "  make smoke-phase07   - Run Phase 07 production hardening smoke test"
	@echo "  make benchmark       - Run multi-model benchmark evaluation on Adult Income"
	@echo "  make ablations       - Run research ablation experiments (E1, E5, E6, E7)"
	@echo "  make report          - Generate docs/PHASE_05_EVALUATION_REPORT.md"
	@echo "  make run-api         - Launch FastAPI REST service on port 8000"
	@echo "  make run-app         - Launch Streamlit interactive dashboard on port 8501"
	@echo "  make backup          - Perform verified online SQLite database backup"
	@echo "  make check-config    - Validate environment and runtime configuration"
	@echo "  make docker-build    - Build multi-stage production Docker container image"
	@echo "  make docker-up       - Launch API and Dashboard via Docker Compose"
	@echo "  make docker-down     - Stop Docker Compose services"
	@echo "  make clean           - Remove cached bytecode and test artifacts"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[full]"

test:
	$(PYTHON) -m pytest tests/ -v

smoke:
	$(PYTHON) scripts/run_phase05_smoke.py

smoke-phase06:
	$(PYTHON) scripts/run_phase06_smoke.py

smoke-phase07:
	$(PYTHON) scripts/run_phase07_smoke.py

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

backup:
	$(PYTHON) scripts/backup_db.py

check-config:
	$(PYTHON) -c "from api.config import settings; print(f'Config OK: env={settings.env}, db={settings.db_path}, origins={settings.allowed_origins}')"

docker-build:
	docker build -t tabsynth-ai:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/ build/ dist/ *.egg-info
