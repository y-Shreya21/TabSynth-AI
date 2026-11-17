# TabSynth AI Deployment & Infrastructure Guide

This guide covers deployment strategies for **TabSynth AI**, contrasting local development on Apple Silicon with containerized production deployment.

---

## 1. Local Development (macOS Apple Silicon M4)

For maximum training and sampling throughput during local development, run natively on macOS to leverage Apple Metal Performance Shaders (MPS):

### Prerequisites
- macOS 14+ (Darwin 23+)
- Python 3.11
- Apple Silicon (M1/M2/M3/M4)

### Native Setup
```bash
# 1. Clone repository
git clone https://github.com/y-Shreya21/TabSynth-AI.git
cd TabSynth-AI

# 2. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install package with full dependencies
pip install --upgrade pip setuptools wheel
pip install -e ".[full]"

# 4. Verify MPS acceleration
python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"
```

### Starting Services Locally
In separate terminal tabs:
```bash
# Terminal 1: FastAPI REST Service (Port 8000)
make run-api

# Terminal 2: Streamlit Dashboard (Port 8501)
make run-app
```

---

## 2. Containerized Deployment (Docker & Docker Compose)

Docker is the recommended deployment method for cloud instances (AWS EC2, GCP Compute Engine, Kubernetes) or staging servers.

> [!IMPORTANT]
> **macOS Docker Limitation:**  
> When running Docker on macOS, containers execute inside a Linux virtual machine. Apple MPS GPU acceleration is **not** accessible inside Linux containers on macOS. On macOS, the Docker container automatically defaults to multi-threaded CPU execution. On Linux hosts equipped with NVIDIA GPUs, PyTorch CUDA acceleration is supported by configuring the NVIDIA Container Toolkit.

### Building and Running with Docker Compose
```bash
# 1. Build container image
docker compose build

# 2. Launch FastAPI backend and Streamlit dashboard
docker compose up -d

# 3. Inspect container logs
docker compose logs -f

# 4. Verify health status
curl -f http://localhost:8000/health
```

### Accessing Endpoints
- **Streamlit Dashboard**: `http://localhost:8501`
- **FastAPI REST API**: `http://localhost:8000`
- **Interactive OpenAPI Documentation**: `http://localhost:8000/docs`

### Persistent Volumes
The `docker-compose.yml` mounts local host directories into `/app`:
- `./data:/app/data`: Uploaded datasets and SQLite database (`tabsynth.db`).
- `./checkpoints:/app/checkpoints`: Serialized PyTorch model weights.
- `./experiments/results:/app/experiments/results`: Serialized benchmark JSON reports.

### Stopping Services
```bash
docker compose down
```
