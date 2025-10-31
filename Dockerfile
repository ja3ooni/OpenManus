# Multi-stage build for production-ready OpenManus container
# Stage 1: Build dependencies and prepare application
FROM python:3.12-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/workspace /app/config /app/data

# Stage 2: Production runtime
FROM python:3.12-slim as production

# Set build arguments for labels
ARG BUILD_DATE
ARG VERSION=latest
ARG VCS_REF

# Set metadata labels
LABEL maintainer="OpenManus Team" \
      org.opencontainers.image.title="OpenManus" \
      org.opencontainers.image.description="Production-ready AI agent framework" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/OpenManus/OpenManus"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r openmanus && useradd -r -g openmanus -d /app -s /bin/bash openmanus

# Set working directory
WORKDIR /app

# Copy Python environment from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code with proper ownership
COPY --from=builder --chown=openmanus:openmanus /app .

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/workspace /app/config /app/data \
    && chown -R openmanus:openmanus /app \
    && chmod -R 755 /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENMANUS_ENV=production \
    OPENMANUS_CONFIG_DIR=/app/config \
    OPENMANUS_WORKSPACE_DIR=/app/workspace \
    OPENMANUS_LOG_DIR=/app/logs \
    OPENMANUS_DATA_DIR=/app/data

# Expose ports
EXPOSE 8000 8080 9090

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

# Switch to non-root user
USER openmanus

# Use tini for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Set default command
CMD ["python", "-m", "app.server"]

# Development stage for local development
FROM builder as development

# Install development dependencies
RUN uv pip install --system --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy

# Set development environment
ENV OPENMANUS_ENV=development \
    PYTHONPATH=/app

# Keep root user for development convenience
USER root

# Default command for development
CMD ["python", "main.py"]
