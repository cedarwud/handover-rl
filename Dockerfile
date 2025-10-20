# Handover-RL V2.0 Docker Image
# Multi-stage build for optimized image size

# ============================================================================
# Stage 1: Builder - Install dependencies
# ============================================================================
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal image
# ============================================================================
FROM python:3.10-slim

# Metadata
LABEL maintainer="Handover-RL Team"
LABEL description="Handover-RL V2.0: RL-based LEO Satellite Handover Optimization"
LABEL version="2.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set PATH to use venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data/episodes \
    /app/checkpoints \
    /app/logs \
    /app/config

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/
COPY tests/ /app/tests/
COPY README.md /app/
COPY FRAMEWORK_STATUS.md /app/

# Copy orbit-engine integration (if available)
# NOTE: orbit-engine should be mounted as volume or copied separately
# COPY ../orbit-engine /app/orbit-engine

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash handover && \
    chown -R handover:handover /app

# Switch to non-root user
USER handover

# Expose ports
# 6006: TensorBoard
# 8888: Jupyter (if enabled)
EXPOSE 6006 8888

# Set working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import gymnasium" || exit 1

# Default command: Python shell
CMD ["/bin/bash"]
