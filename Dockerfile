# syntax=docker/dockerfile:1

# ---- Stage 1: build ----------------------------------------------------------------
# Python 3.12 slim — TensorFlow 2.20 wheels are published for 3.12 (not 3.13+).
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and from buffering stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System libraries needed by TensorFlow + Pillow at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching).
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the rest of the application code.
COPY . .

# The service listens on this port (Cloud Run injects $PORT at runtime).
ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
