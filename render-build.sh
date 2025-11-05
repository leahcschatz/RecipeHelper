#!/usr/bin/env bash
# Exit on first error
set -o errexit  

# --- Install system dependencies ---
apt-get update && apt-get install -y \
  poppler-utils \        # for PDF → image conversion
  tesseract-ocr \        # for OCR text extraction
  libtesseract-dev \     # for pytesseract compatibility
  libleptonica-dev       # sometimes required by tesseract

# --- Install Python dependencies ---
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo "✅ System and Python dependencies installed successfully."
