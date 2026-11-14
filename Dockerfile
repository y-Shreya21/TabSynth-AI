# ==============================================================================
# TabSynth AI - Multi-Stage Production Dockerfile
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /install

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt pyproject.toml ./

# Install CPU PyTorch and dependencies into wheels / prefix
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Production Runtime
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/tabsynth/.local/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH"

# Install curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create secure non-root user
RUN groupadd -g 10001 tabsynth && \
    useradd -u 10001 -g tabsynth -m -s /bin/bash tabsynth

WORKDIR /app

# Copy installed python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source code
COPY --chown=tabsynth:tabsynth src/ /app/src/
COPY --chown=tabsynth:tabsynth api/ /app/api/
COPY --chown=tabsynth:tabsynth app/ /app/app/
COPY --chown=tabsynth:tabsynth configs/ /app/configs/
COPY --chown=tabsynth:tabsynth scripts/ /app/scripts/
COPY --chown=tabsynth:tabsynth experiments/ /app/experiments/
COPY --chown=tabsynth:tabsynth pyproject.toml /app/pyproject.toml

# Install tabsynth in editable mode for non-root user
RUN pip install --no-deps -e /app

# Create persistent storage directories with tabsynth permissions
RUN mkdir -p /app/data/raw /app/data/processed /app/data/synthetic /app/checkpoints /app/experiments/results && \
    chown -R tabsynth:tabsynth /app

USER tabsynth

# Default ports: 8000 (FastAPI), 8501 (Streamlit)
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
