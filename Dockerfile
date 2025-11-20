# Use a lightweight Python base image
FROM python:3.12-slim

# Install system dependencies for tesseract, pdf2image, and build tools
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better Docker caching)
COPY requirements.txt .

# Upgrade pip first
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Tell Flask / gunicorn which app to run
ENV FLASK_APP=app.py

# Expose the port Render will use internally
EXPOSE 10000

ENV WEB_CONCURRENCY=1
ENV GUNICORN_CMD_ARGS="--timeout 180 --workers 1 --threads 1"

# Start the app with gunicorn
# Render will set PORT env var, so we use that
CMD exec gunicorn --bind 0.0.0.0:${PORT:-10000} app:app
