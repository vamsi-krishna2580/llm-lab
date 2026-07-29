#!/bin/bash
set -e

echo "======================================"
echo "Setting up LLM Lab..."
echo "======================================"

export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH

echo "Updating packages..."
apt update -y

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "======================================"
echo "Starting API..."
echo "======================================"

python -m uvicorn chat:app --host 0.0.0.0 --port 8000