#!/bin/bash
#SBATCH --job-name=p9-gate0-pilot
#SBATCH --output=logs/p9_gate0_%j.out
#SBATCH --error=logs/p9_gate0_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Plan9 Gate 0 SMOKE PILOT: fused subs+frame window oracle on Video-MME-long.
# Serves Qwen2.5-VL-7B, runs scripts/window_oracle_fused.py at a small limit.
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs reports

export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLM_MAX_PIXELS="${VLM_MAX_PIXELS:-200704}"
export VLM_MIN_PIXELS="${VLM_MIN_PIXELS:-3136}"

PY="$HOME/miniconda3/envs/vidcurator/bin/python"
MODEL_PATH="models/qwen2_5_vl_7b"
PORT="${PORT:-8017}"
LIMIT="${LIMIT:-24}"
OUT="${OUT:-reports/window_oracle_fused_videomme_pilot.md}"

echo "=== [$SLURM_JOB_ID] plan9 gate0 fused pilot $(date) ==="
echo "Node: $(hostname) | limit=$LIMIT | port=$PORT"
$PY -c "import torch; assert torch.cuda.is_available(); print('cuda OK:', torch.cuda.get_device_name(0))"

cleanup() { [[ -n "${SERVER_PID:-}" ]] && { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }; }
trap cleanup EXIT

echo "--- Serving $MODEL_PATH on $PORT ---"
CUDA_VISIBLE_DEVICES=0 $PY scripts/serve_transformers_vlm.py \
    --model "$MODEL_PATH" --port "$PORT" --dtype bfloat16 \
    --max-pixels "$VLM_MAX_PIXELS" --min-pixels "$VLM_MIN_PIXELS" \
    > "logs/vlm_p9_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!

echo "Waiting for VLM..."
ready=0
for attempt in $(seq 1 180); do
    if $PY -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5)" 2>/dev/null; then
        ready=1; echo "VLM ready after $((attempt*5))s"; break
    fi
    sleep 5
done
if [[ "$ready" -ne 1 ]]; then echo "ERROR: VLM failed to start" >&2; tail -30 "logs/vlm_p9_${SLURM_JOB_ID}.log" >&2; exit 1; fi
if grep -q "Model loaded on cpu" "logs/vlm_p9_${SLURM_JOB_ID}.log"; then echo "ERROR: model on CPU" >&2; exit 1; fi

echo "--- Running fused window oracle (limit=$LIMIT) ---"
$PY scripts/window_oracle_fused.py \
    --qa data/processed/videomme_long_450.jsonl \
    --frames data/indices/videomme_long_decoded_frames.jsonl \
    --subtitle-dir data/subtitles/videomme \
    --endpoint "http://localhost:${PORT}" \
    --windows 8 --per-window 8 --subtitle-chars 3500 \
    --limit "$LIMIT" --out "$OUT" \
    2>&1 | tee "logs/p9_oracle_${SLURM_JOB_ID}.log"

echo "=== [$SLURM_JOB_ID] done $(date) — see $OUT ==="
