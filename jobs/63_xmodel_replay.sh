#!/bin/bash
#SBATCH --output=logs/xmodel_%x_%j.out
#SBATCH --error=logs/xmodel_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# L4 ledger-portability replay: serve MODEL_PATH, replay stored buffers via
# scripts/eval_cross_model.py --with-ledger.
# Env: MODEL_PATH, PORT, PREDS, QA, FRAMES, OUT, POLICIES (default dde_curator),
#      BUDGETS (default 8), SERVER_PY (optional alt python for the server).
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs runs reports

export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLM_MAX_PIXELS="${VLM_MAX_PIXELS:-200704}"
export PYTHONUNBUFFERED=1

PORT="${PORT:-8545}"
POLICIES="${POLICIES:-dde_curator}"
BUDGETS="${BUDGETS:-8}"

cleanup() { [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT
"${SERVER_PY:-python}" scripts/serve_transformers_vlm.py --model "$MODEL_PATH" --port "$PORT" \
  --dtype bfloat16 > "logs/vlm_${SLURM_JOB_ID}_xmodel.log" 2>&1 &
SERVER_PID=$!
ready=0
for attempt in $(seq 1 180); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5)" 2>/dev/null; then
    ready=1; echo "VLM ready (${attempt})"; break
  fi
  sleep 5
done
[[ "$ready" -eq 1 ]] || { echo "ERROR: VLM failed" >&2; tail -30 "logs/vlm_${SLURM_JOB_ID}_xmodel.log" >&2; exit 1; }

python scripts/eval_cross_model.py --preds "$PREDS" --qa "$QA" --frames "$FRAMES" \
  --endpoint "http://127.0.0.1:${PORT}" --policies $POLICIES --budgets $BUDGETS \
  --with-ledger --out "$OUT" 2>&1 | tee "logs/xmodel_eval_${SLURM_JOB_ID}.log"
echo "=== [$SLURM_JOB_ID] xmodel replay done ==="
