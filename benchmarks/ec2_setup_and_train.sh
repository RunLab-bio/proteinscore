#!/bin/bash
# EC2 GPU Training Setup Script
# For g5 (A10G) or g4dn (T4) instances
#
# Usage:
#   1. Launch EC2 spot instance (g5.xlarge or g4dn.xlarge)
#   2. SSH into instance
#   3. Clone repo and run this script
#
# Recommended AMI: Deep Learning AMI GPU PyTorch 2.x (Ubuntu 22.04)

set -e

echo "=========================================="
echo "ESM-2 HIC Distillation - EC2 GPU Setup"
echo "=========================================="

# Check GPU
echo ""
echo "Checking GPU..."
nvidia-smi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install transformers fair-esm scikit-learn

# Verify CUDA
echo ""
echo "Verifying PyTorch CUDA..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Run training
echo ""
echo "=========================================="
echo "Starting Training..."
echo "=========================================="

cd /home/ubuntu/ProteinScore  # Adjust path as needed

# Option 1: ESM-2 650M (best quality, needs ~8GB VRAM)
# Works on: g5.xlarge (24GB), g4dn.xlarge (16GB)
python benchmarks/esm_hic_distillation_gpu.py \
    --teacher facebook/esm2_t33_650M_UR50D \
    --epochs 300 \
    --alpha 0.3 \
    --save-model models/hic_distilled_650m.pt

# Option 2: ESM-2 150M (faster, needs ~4GB VRAM)
# Uncomment if 650M doesn't fit
# python benchmarks/esm_hic_distillation_gpu.py \
#     --teacher facebook/esm2_t30_150M_UR50D \
#     --epochs 300 \
#     --alpha 0.3 \
#     --save-model models/hic_distilled_150m.pt

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Model saved to: models/hic_distilled_*.pt"
echo "Copy this file back to your local machine for deployment"
