# TabSynth Analytics: Production Deployment & Operations Guide

This guide describes how to configure, deploy, operate, and maintain **TabSynth Analytics** in production environments.

---

## 1. System Requirements & Architecture

### Recommended Hardware
- **CPU**: 4+ Cores (ARM64 or x86_64)
- **Memory**: 8 GB minimum (16 GB recommended for high-dimensional tabular training)
- **Disk**: 20 GB SSD storage for checkpoints, raw datasets, and evaluation artifacts
- **GPU / Acceleration**:
  - **Local Development**: Apple Silicon (M1/M2/M3/M4) via Apple MPS acceleration, or NVIDIA CUDA (Linux/Windows).
  - **Docker Containers**: Standard CPU fallback is used inside Linux containers. (MPS is not accessible inside Linux Docker containers).

### Network Ports
- **FastAPI Backend**: `8000` (HTTP REST API, OpenAPI docs at `/docs`)
- **Streamlit Frontend**: `8501` (HTTP & WebSocket UI)

---

## 2. Local Installation & Startup

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/y-Shreya21/TabSynth-AI.git
cd "TabSynth AI"

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[full]"
```

### Step 2: Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` to configure your settings:
```bash
TABSYNTH_ENV=production
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
TABSYNTH_API_URL=http://127.0.0.1:8000
CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
MAX_CONCURRENT_JOBS=2
```

### Step 3: Run Services Locally
In Terminal 1 (FastAPI Backend):
```bash
make run-api
# or: uvicorn api.server:app --host 0.0.0.0 --port 8000
```

In Terminal 2 (Streamlit Frontend):
```bash
make run-app
# or: streamlit run app/main.py
```

---

## 3. Containerized Deployment with Docker Compose

TabSynth AI provides a production-grade multi-container architecture using `docker-compose.yml`.

### Step 1: Verify Configuration
Ensure `.env` contains the required environment variables:
```bash
TABSYNTH_ENV=production
TABSYNTH_DEVICE=cpu
MAX_CONCURRENT_JOBS=2
```

### Step 2: Build and Start Containers
```bash
# Build images and start containers in detached mode
make docker-up
# or: docker compose up -d --build
```

### Step 3: Verify Container Health
```bash
docker compose ps
```
Both `tabsynth-api` and `tabsynth-dashboard` should report status `healthy`.

Health check endpoints:
- API Liveness: `curl -f http://localhost:8000/health`
- API Readiness: `curl -f http://localhost:8000/ready`
- Dashboard Health: `curl -f http://localhost:8501/_stcore/health`

### Step 4: Stop Containers
```bash
make docker-down
# or: docker compose down
```

---

## 4. Reverse Proxy Setup (Nginx)

For public technical demonstrations or production deployment behind TLS/SSL, configure Nginx as a reverse proxy:

```nginx
# /etc/nginx/conf.d/tabsynth.conf

# FastAPI Backend
server {
    listen 80;
    server_name api.tabsynth.example.com;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Streamlit Frontend Dashboard
server {
    listen 80;
    server_name dashboard.tabsynth.example.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (essential for Streamlit interactive stream)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

---

## 5. Database Backup, Restore & Maintenance

### Automated Online Backup
Run the backup script without interrupting active reads or writes:
```bash
make backup
# or: python scripts/backup_db.py --dest backups/
```
This utility:
- Uses SQLite's online backup API with WAL mode.
- Validates database integrity (`PRAGMA integrity_check`) after copy.
- Enforces retention rotation (default: keeps latest 7 daily snapshots).

### Database Restore
To restore from a backup archive:
```bash
python scripts/restore_db.py --source backups/tabsynth_backup_20261117_120000.db
```

### Automatic Crash Recovery
Upon restart, TabSynth Analytics executes `recover_orphaned_jobs()`. Any background training or synthesis jobs that were interrupted by a server crash or process termination will be transitioned from `running` or `queued` to `failed` with a clear audit log entry, preventing stale progress displays.

---

## 6. Monitoring & Logging

### Structured Logs
The API outputs JSON-formatted and formatted structured logs containing:
- ISO-8601 timestamp
- Log severity (`INFO`, `WARNING`, `ERROR`)
- Correlation Request ID (`[ReqID: ...]`)
- HTTP method, route, response status, and duration in milliseconds

To follow logs in Docker:
```bash
docker compose logs -f api
docker compose logs -f dashboard
```
