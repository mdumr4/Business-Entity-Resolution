#!/usr/bin/env bash
# ==============================================================================
# GCP Vertex AI Workbench / Compute Engine Provisioning Script
# ==============================================================================
# Target Hardware:
#   - Machine Type: g2-standard-16 (16 vCPU, 64 GB RAM)
#   - Accelerator: 1x NVIDIA L4 GPU (24 GB VRAM)
#   - Disk: 500 GB Balanced Persistent Disk (pd-balanced)
#   - OS: Ubuntu 22.04 LTS / Deep Learning VM (CUDA 12.x)
# ==============================================================================

set -euo pipefail

echo "=========================================================="
echo " Setting up Entity Resolution Cloud Environment on GCP..."
echo "=========================================================="

# 1. System packages
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    curl \
    wget \
    htop \
    tmux \
    python3-dev \
    python3-pip \
    python3-venv

git lfs install

# 2. Check NVIDIA GPU & Drivers
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA Driver detected:"
    nvidia-smi
else
    echo "WARNING: nvidia-smi not found. Ensure NVIDIA CUDA drivers are installed."
fi

# 3. Create and activate Python virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip setuptools wheel

# 4. Install dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# 5. Create required output directories
mkdir -p output/faiss_train output/faiss_test output/eval dataset/train dataset/test

echo "=========================================================="
echo " GCP Environment Setup Complete!"
echo " Activate with: source venv/bin/activate"
echo "=========================================================="
