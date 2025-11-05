#!/usr/bin/env bash
# render-build.sh
set -o errexit  # exit on first error

echo "Installing system packages..."
apt-get update && apt-get install -y tesseract-ocr poppler-utils libtesseract-dev libleptonica-dev

echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo "✅ Build completed successfully"
