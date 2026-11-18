# ==============================================================================
# TabSynth Analytics - Hardened Production Multi-Stage Containerfile
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder (Compile dependencies and wheels)
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging specifications
COPY requirements.txt pyproject.toml /build/

# Install dependencies into isolated local environment prefix
RUN pip install --upgrade pip setuptools wheel && \
    pip install --prefix=/install -r requirements.txt && \
    pip install --prefix=/install -e .

# ------------------------------------------------------------------------------
# Stage 2: Production Runtime (Minimal, secure, non-root execution)
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Standard production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TABSYNTH_ENV=production \
    TABSYNTH_DATA_DIR=/app/data \
    TABSYNTH_RAW_DIR=/app/data/raw \
    TABSYNTH_SYNTHETIC_DIR=/app/data/synthetic \
    TABSYNTH_CHECKPOINT_DIR=/app/checkpoints \
    TABSYNTH_EXPERIMENT_DIR=/app/experiments/results \
    TABSYNTH_BACKUP_DIR=/app/backups \
    TABSYNTH_DB_PATH=/app/data/tabsynth.db \
    TABSYNTH_DEVICE=cpu \
    PATH="/install/bin:$PATH" \
    PYTHONPATH="/app:/install/lib/python3.11/site-packages:$PYTHONPATH"

# Install minimal runtime shared libraries (curl for healthchecks, libgomp for PyTorch OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /install

# Create non-root unprivileged service account
RUN groupadd -g 1000 tabsynth && \
    useradd -u 1000 -g tabsynth -m -s /bin/bash tabsynth && \
    mkdir -p /app/data/raw \
             /app/data/processed \
             /app/data/synthetic \
             /app/checkpoints \
             /app/experiments/results \
             /app/backups && \
    chown -R tabsynth:tabsynth /app

# Copy application code into container
COPY --chown=tabsynth:tabsynth . /app

# Switch to unprivileged execution user
USER tabsynth

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Healthcheck probe targeting API liveness
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command starts FastAPI REST service
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
