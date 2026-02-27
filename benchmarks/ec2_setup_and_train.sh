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

# Create models directory
mkdir -p models

# ============================================
# STUDENT MODEL SIZE OPTIONS (for CPU deployment)
# ============================================
# tiny:   ~100K params, ~5ms  CPU latency
# small:  ~300K params, ~10ms CPU latency
# medium: ~800K params, ~20ms CPU latency (balanced)
# large:  ~2M params,   ~50ms CPU latency (highest quality)

# ============================================
# Option 1: ESM-2 650M Teacher + Medium Student
# ============================================
# Best quality teacher, balanced student
# Needs ~8GB VRAM (works on g5.xlarge, g4dn.xlarge)
python benchmarks/esm_hic_distillation_gpu.py \
    --teacher facebook/esm2_t33_650M_UR50D \
    --size medium \
    --epochs 300 \
    --alpha 0.3 \
    --save-model models/hic_distilled_650m_medium.pt

# ============================================
# Option 2: ESM-2 650M Teacher + Tiny Student
# ============================================
# Best teacher, smallest student (fastest CPU)
# Uncomment to train tiny model
# python benchmarks/esm_hic_distillation_gpu.py \
#     --teacher facebook/esm2_t33_650M_UR50D \
#     --size tiny \
#     --epochs 300 \
#     --alpha 0.3 \
#     --save-model models/hic_distilled_650m_tiny.pt

# ============================================
# Option 3: ESM-2 150M Teacher (faster training)
# ============================================
# Needs ~4GB VRAM
# python benchmarks/esm_hic_distillation_gpu.py \
#     --teacher facebook/esm2_t30_150M_UR50D \
#     --size small \
#     --epochs 300 \
#     --alpha 0.3 \
#     --save-model models/hic_distilled_150m_small.pt

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: models/hic_distilled_*.pt"
echo ""
echo "Student Size Options:"
echo "  tiny:   ~100K params, ~5ms  CPU"
echo "  small:  ~300K params, ~10ms CPU"
echo "  medium: ~800K params, ~20ms CPU"
echo "  large:  ~2M params,   ~50ms CPU"
echo ""
echo "Copy model back to local machine:"
echo "  scp ubuntu@<ec2-ip>:~/ProteinScore/models/*.pt ."
