#!/bin/bash

echo "Starting Grounding DINO Multi-Modal Training..."
echo "=============================================="

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_DIR"

echo "Project Directory: $PROJECT_DIR"
echo "Available GPUs:"
nvidia-smi

echo ""
echo "--- Single GPU Training ---"
echo "Command: python tools/train.py"
echo ""

python tools/train.py