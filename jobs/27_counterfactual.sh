#!/bin/bash
#SBATCH --job-name=vc-ei-cf
#SBATCH --output=logs/ei_cf_%j.out
#SBATCH --error=logs/ei_cf_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# E-I counterfactual: serve the VLM, then re-answer nextgqa correct subset with evidence
# frames removed/replaced (scripts/eval_counterfactual.py). Env: PORT, PREDS, QA, FRAMES,
# BUDGET, POLICIES, OUT, LIMIT (0=full), MODEL_PATH.
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs runs reports

export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLM_MAX_PIXELS="${VLM_MAX_PIXELS:-200704}" VLM_MIN_PIXELS="${VLM_MIN_PIXELS:-3136}"

PY="$HOME/miniconda3/envs/vidcurator/bin/python"
MODEL_PATH="${MODEL_PATH:-models/qwen2_5_vl_7b}"
PORT="${PORT:-8090}"
PREDS="${PREDS:-runs/r1_v2_nextgqa_qwen2_5/predictions.jsonl}"
QA="${QA:-data/processed/nextgqa_val_150.jsonl}"
FRAMES="${FRAMES:-data/indices/nextgqa_siglip_frames.jsonl}"
BUDGET="${BUDGET:-8}"
POLICIES="${POLICIES:-uniform aks scripted_curator curator_siglip}"
OUT="${OUT:-reports/ei_counterfactual_nextgqa.md}"
LIMIT="${LIMIT:-0}"

echo "=== [$SLURM_JOB_ID] E-I counterfactual on $(hostname) at $(date) ==="
$PY -c "import torch; assert torch.cuda.is_available(); print('cuda OK:', torch.cuda.get_device_name(0))"

cleanup() { [[ -n "${SERVER_PID:-}" ]] && { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }; }
trap cleanup EXIT

SERVE_LOG="logs/vlm_ei_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=0 $PY scripts/serve_transformers_vlm.py \
  --model "$MODEL_PATH" --port "$PORT" --dtype bfloat16 \
  --max-pixels "$VLM_MAX_PIXELS" --min-pixels "$VLM_MIN_PIXELS" > "$SERVE_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for VLM..."
ready=0
for attempt in $(seq 1 180); do
  if $PY -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5)" 2>/dev/null; then
    ready=1; echo "VLM ready ($(( attempt * 5 ))s)"; break
  fi
  sleep 5
done
[[ "$ready" -ne 1 ]] && { echo "ERROR: VLM failed to start" >&2; tail -30 "$SERVE_LOG" >&2; exit 1; }

$PY scripts/eval_counterfactual.py \
  --preds "$PREDS" --qa "$QA" --frames "$FRAMES" \
  --endpoint "http://127.0.0.1:${PORT}" --budget "$BUDGET" \
  --policies $POLICIES --out "$OUT" --limit "$LIMIT"

echo "=== [$SLURM_JOB_ID] E-I done at $(date) ==="
