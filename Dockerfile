# Multi-stage build to reduce final image size
# Stage 1: Build dependencies
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime image (much smaller)
FROM python:3.12-slim

# Install only runtime system dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Tell Flask / gunicorn which app to run
ENV FLASK_APP=app.py

# Expose the port
EXPOSE 10000

ENV WEB_CONCURRENCY=1
ENV GUNICORN_CMD_ARGS="--timeout 180 --workers 1 --threads 1"

# Start the app with gunicorn
CMD exec gunicorn --bind 0.0.0.0:${PORT:-10000} app:app
