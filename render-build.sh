#!/usr/bin/env bash
set -o errexit  # stop on first error

# Install system dependencies (Ubuntu)
apt-get update && apt-get install -y \
  poppler-utils \
  tesseract-ocr \
  libtesseract-dev \
  libleptonica-dev

# Verify installation (for Render logs)
echo "Tesseract version:"
tesseract --version || echo "Tesseract not found after install!"

# Install Python dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo "✅ Build completed successfully!"
