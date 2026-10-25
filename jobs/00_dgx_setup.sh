#!/bin/bash
#SBATCH --job-name=vidcurator-dgx-setup
#SBATCH --output=/home/vis-comp/22b3929/vidcurator/logs/dgx_setup_%j.out
#SBATCH --error=/home/vis-comp/22b3929/vidcurator/logs/dgx_setup_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator

echo "=== Setting up venv on DGX node $(hostname) ==="
echo "System python: $(python3 --version 2>&1)"

VENV_DIR="/home/vis-comp/22b3929/vidcurator/.venv_dgx"

# Use virtualenv (doesn't need ensurepip)
/home/vis-comp/22b3929/vidcurator/.venv/bin/virtualenv --python=python3 --copies "$VENV_DIR"
PY="$VENV_DIR/bin/python"
echo "Venv python: $($PY --version 2>&1)"

# Install pip manually if needed
$PY -m ensurepip 2>/dev/null || curl -sS https://bootstrap.pypa.io/get-pip.py | $PY 2>/dev/null || {
    echo "Installing pip via get-pip..."
    wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py
    $PY /tmp/get-pip.py
}

$PY -m pip install --upgrade pip

echo "Installing GPU deps (this takes ~15 min)..."
$PY -m pip install \
    "torch>=2.4.0" \
    "transformers>=4.57.0" \
    "accelerate>=1.0.0" \
    "qwen-vl-utils>=0.0.8" \
    "sentence-transformers>=3.0.0" \
    "pillow>=10.0.0" \
    "huggingface-hub>=0.26.0" \
    "imageio-ffmpeg>=0.5.1" \
    "pandas>=2.0.0" \
    "pyarrow>=14.0.0" \
    "pytest>=8.0.0"

$PY -m pip install -e .

echo "=== Verifying ==="
$PY -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
$PY -c "import transformers; print(f'transformers {transformers.__version__}')"
echo "=== DGX setup done ==="
