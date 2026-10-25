#!/bin/bash
#SBATCH --job-name=vc-cuda-test
#SBATCH --output=logs/cuda_test_%j.out
#SBATCH --error=logs/cuda_test_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
python -c "
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    x = torch.randn(2,2, device='cuda')
    print('matmul_ok', (x@x).sum().item())
"
