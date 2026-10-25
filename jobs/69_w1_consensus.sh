#!/bin/bash
#SBATCH --output=logs/w1cons_%j.out
#SBATCH --error=logs/w1cons_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# W1/W2: consensus + reliability-tagged ledger replay (3 arms, READ held fixed).
# Env: VARIANTS, QA, FRAMES, OUT, PORT.
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
  > "logs/vlm_${SLURM_JOB_ID}_w1cons.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null' EXIT
for attempt in $(seq 1 180); do
  python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5)" 2>/dev/null && break
  sleep 5
done

python scripts/w1_consensus.py --variants "$VARIANTS" --qa "$QA" --frames "$FRAMES" \
  --endpoint "http://127.0.0.1:${PORT}" --k 8 --out "$OUT"
