# TabSynth AI Production Multi-Service Container
# Python 3.11 Base Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TABSYNTH_DATA_DIR=/app/data \
    TABSYNTH_CHECKPOINT_DIR=/app/checkpoints \
    TABSYNTH_EXPERIMENT_DIR=/app/experiments/results \
    TABSYNTH_DB_PATH=/app/data/tabsynth.db

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first for layer caching
COPY requirements.txt pyproject.toml /app/
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -e .

# Create non-root application user
RUN useradd -m -u 1000 tabsynth && \
    mkdir -p /app/data/raw /app/data/processed /app/data/synthetic /app/checkpoints /app/experiments/results && \
    chown -R tabsynth:tabsynth /app

# Copy application source code
COPY --chown=tabsynth:tabsynth . /app

# Switch to non-root user
USER tabsynth

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Default healthcheck targeting FastAPI probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default entrypoint launches FastAPI backend
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
