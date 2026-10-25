#!/bin/bash
#SBATCH --job-name=vc-xfam
#SBATCH --output=logs/xfam_%j.out
#SBATCH --error=logs/xfam_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
# E-G cross-FAMILY portability: replay the Qwen2.5 curator's selected frames to an InternVL3.5-8B
# answerer (different model family). Tests whether the curated buffer is portable across families,
# not just across sizes. No Qwen pixel flags (InternVL uses its own 448px tile transform).
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs reports
export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="$HOME/miniconda3/envs/vidcurator/bin/python"
MODEL_PATH="${MODEL_PATH:-models/internvl3_5_8b}"    # cross-family answerer
PORT="${PORT:-8096}"
PREDS="${PREDS:?PREDS required}"
QA="${QA:?QA required}"
FRAMES="${FRAMES:?FRAMES required}"
POLICIES="${POLICIES:-uniform curator_siglip scripted_curator}"
BUDGETS="${BUDGETS:-8}"
OUT="${OUT:?OUT required}"
LIMIT="${LIMIT:-500}"
echo "=== [$SLURM_JOB_ID] E-G cross-FAMILY on $(hostname) at $(date), answerer=$MODEL_PATH ==="
$PY -c "import torch; assert torch.cuda.is_available(); print('cuda OK:', torch.cuda.get_device_name(0))"
cleanup(){ [[ -n "${SERVER_PID:-}" ]] && { kill "$SERVER_PID" 2>/dev/null||true; wait "$SERVER_PID" 2>/dev/null||true; }; }
trap cleanup EXIT
SERVE_LOG="logs/vlm_xfam_${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=0 $PY scripts/serve_transformers_vlm.py --model "$MODEL_PATH" --port "$PORT" --dtype bfloat16 \
  > "$SERVE_LOG" 2>&1 &
SERVER_PID=$!
ready=0
for a in $(seq 1 240); do
  if $PY -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5)" 2>/dev/null; then ready=1; echo "VLM ready ($((a*5))s)"; break; fi
  sleep 5
done
[[ "$ready" -ne 1 ]] && { echo "ERROR: VLM failed" >&2; tail -40 "$SERVE_LOG" >&2; exit 1; }
if grep -q "Model loaded on cpu" "$SERVE_LOG"; then echo "ERROR: model on CPU" >&2; exit 1; fi
$PY scripts/eval_cross_model.py --preds "$PREDS" --qa "$QA" --frames "$FRAMES" \
  --endpoint "http://127.0.0.1:${PORT}" --policies $POLICIES --budgets $BUDGETS --out "$OUT" --limit "$LIMIT"
echo "=== [$SLURM_JOB_ID] E-G cross-family done at $(date) -> $OUT ==="
