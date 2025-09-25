#!/bin/bash

# AWS GPU Instance Setup Script
# Run this after connecting to your AWS g4dn.xlarge instance
#
# Usage: bash aws_setup.sh

set -e

echo "=== AWS GPU Instance Setup for CSI Research ==="
echo

# Check if we're on AWS
if [ -f /sys/hypervisor/uuid ] && [ "$(head -c 3 /sys/hypervisor/uuid)" = "ec2" ]; then
    echo "✓ Running on AWS EC2 instance"
else
    echo "⚠️  Warning: This doesn't appear to be an AWS instance"
fi

# Check GPU
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✓ NVIDIA GPU detected"
else
    echo "❌ No GPU detected. Make sure you're on a g4dn.xlarge instance"
    exit 1
fi

# Check if conda is available (Deep Learning AMI should have it)
if command -v conda &> /dev/null; then
    echo "✓ Conda detected"
    
    # Activate PyTorch environment
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate pytorch
    echo "✓ PyTorch environment activated"
    
    # Check PyTorch GPU support
    python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
else
    echo "⚠️  Conda not found. You may need to set up Python environment manually."
fi

echo
echo "=== Install Additional Dependencies ==="

# Install missing packages that aren't in the Deep Learning AMI
pip install scikit-learn>=1.3.0 tqdm>=4.64.0

echo "✓ Additional dependencies installed"

echo
echo "=== Instance Setup Complete ==="
echo
echo "Next steps:"
echo "1. Upload your code:"
echo "   scp -i your-key.pem -r ./csi-diss-25 ec2-user@$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):/home/ec2-user/"
echo
echo "2. Run experiments:"
echo "   cd csi-diss-25"
echo "   bash run_experiments_gpu.sh"
echo
echo "3. Monitor GPU usage:"
echo "   watch nvidia-smi"
echo
echo "4. Download results when done:"
echo "   scp -i your-key.pem -r ec2-user@$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):/home/ec2-user/csi-diss-25/experiments ./results/"
echo
echo "Instance details:"
echo "  Public IP: $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo "  Instance Type: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)"
echo "  Region: $(curl -s http://169.254.169.254/latest/meta-data/placement/region)"