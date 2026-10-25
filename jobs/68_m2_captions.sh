#!/bin/bash
#SBATCH --output=logs/m2cap_%j.out
#SBATCH --error=logs/m2cap_%j.err
#SBATCH --time=22:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# M2 phase 1: caption-variant generation + retention-utility scoring + headroom GATE.
# Env: QA, FRAMES, OUT, PORT.
set -uo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs reports
export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLM_MAX_PIXELS="${VLM_MAX_PIXELS:-200704}"
export PYTHONUNBUFFERED=1

python scripts/serve_transformers_vlm.py --model models/qwen2_5_vl_7b --port "$PORT" --dtype bfloat16 \
  > "logs/vlm_${SLURM_JOB_ID}_m2cap.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null' EXIT
for attempt in $(seq 1 180); do
  python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5)" 2>/dev/null && break
  sleep 5
done

python scripts/rft_captions.py --qa "$QA" --frames "$FRAMES" \
  --endpoint "http://127.0.0.1:${PORT}" --variants 6 --videos-cap 350 --out "$OUT"
