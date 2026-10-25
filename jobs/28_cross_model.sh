#!/bin/bash
#SBATCH --job-name=vc-eg-xmodel
#SBATCH --output=logs/eg_xmodel_%j.out
#SBATCH --error=logs/eg_xmodel_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs reports
export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLM_MAX_PIXELS="${VLM_MAX_PIXELS:-200704}" VLM_MIN_PIXELS="${VLM_MIN_PIXELS:-3136}"
PY="$HOME/miniconda3/envs/vidcurator/bin/python"
MODEL_PATH="${MODEL_PATH:-models/qwen3_vl_8b_instruct}"   # cross-answerer (Qwen3)
PORT="${PORT:-8095}"
PREDS="${PREDS:-runs/r1_v2_nextgqa_qwen2_5/predictions.jsonl}"
QA="${QA:-data/processed/nextgqa_val_150.jsonl}"
FRAMES="${FRAMES:-data/indices/nextgqa_siglip_frames.jsonl}"
POLICIES="${POLICIES:-uniform curator_siglip}"
BUDGETS="${BUDGETS:-8}"
OUT="${OUT:-reports/eg_cross_model_nextgqa.md}"
LIMIT="${LIMIT:-0}"
echo "=== [$SLURM_JOB_ID] E-G cross-model on $(hostname) at $(date), answerer=$MODEL_PATH ==="
$PY -c "import torch; assert torch.cuda.is_available(); print('cuda OK:', torch.cuda.get_device_name(0))"
cleanup(){ [[ -n "${SERVER_PID:-}" ]] && { kill "$SERVER_PID" 2>/dev/null||true; wait "$SERVER_PID" 2>/dev/null||true; }; }
trap cleanup EXIT
SERVE_LOG="logs/vlm_eg_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=0 $PY scripts/serve_transformers_vlm.py --model "$MODEL_PATH" --port "$PORT" --dtype bfloat16 \
  --max-pixels "$VLM_MAX_PIXELS" --min-pixels "$VLM_MIN_PIXELS" > "$SERVE_LOG" 2>&1 &
SERVER_PID=$!
ready=0
for a in $(seq 1 240); do
  if $PY -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5)" 2>/dev/null; then ready=1; echo "VLM ready ($((a*5))s)"; break; fi
  sleep 5
done
[[ "$ready" -ne 1 ]] && { echo "ERROR: VLM failed" >&2; tail -30 "$SERVE_LOG" >&2; exit 1; }
$PY scripts/eval_cross_model.py --preds "$PREDS" --qa "$QA" --frames "$FRAMES" \
  --endpoint "http://127.0.0.1:${PORT}" --policies $POLICIES --budgets $BUDGETS --out "$OUT" --limit "$LIMIT"
echo "=== [$SLURM_JOB_ID] E-G done at $(date) ==="
