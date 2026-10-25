#!/bin/bash
#SBATCH --job-name=vc-pilot-multi
#SBATCH --output=logs/pilot_multi_%x_%j.out
#SBATCH --error=logs/pilot_multi_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# VID-MEM pilot runner: serve ONE VLM, evaluate MULTIPLE configs sequentially.
# Required env: CONFIGS (space-separated config paths). Optional: MODEL_PATH, PORT.
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs runs

export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# GPUS_VISIBLE=all keeps Slurm's own CUDA_VISIBLE_DEVICES (all allocated GPUs, for
# VLM_DEVICE_MAP=auto sharding); sbatch --export cannot pass comma lists like "0,1".
if [[ "${GPUS_VISIBLE:-0}" != "all" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPUS_VISIBLE:-0}"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLM_MAX_PIXELS="${VLM_MAX_PIXELS:-200704}"
export PYTHONUNBUFFERED=1

MODEL_PATH="${MODEL_PATH:-models/qwen2_5_vl_7b}"
PORT="${PORT:-8000}"
: "${CONFIGS:?CONFIGS env var required (space-separated config paths)}"

cleanup() { [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "=== [$SLURM_JOB_ID] pilot-multi on $(hostname): $CONFIGS ==="
# SERVER_PY: alternate python for the VLM server only (evals stay on vidcurator env).
# Needed for models whose transformers path requires torch>=2.6 (e.g. gemma3 masks).
"${SERVER_PY:-python}" scripts/serve_transformers_vlm.py --model "$MODEL_PATH" --port "$PORT" --dtype bfloat16 \
  > "logs/vlm_${SLURM_JOB_ID}_pilot.log" 2>&1 &
SERVER_PID=$!

ready=0
for attempt in $(seq 1 180); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5)" 2>/dev/null; then
    ready=1; echo "VLM ready (${attempt} attempts)"; break
  fi
  sleep 5
done
[[ "$ready" -eq 1 ]] || { echo "ERROR: VLM failed to start" >&2; tail -30 "logs/vlm_${SLURM_JOB_ID}_pilot.log" >&2; exit 1; }
grep -q "Model loaded on cpu" "logs/vlm_${SLURM_JOB_ID}_pilot.log" && { echo "ERROR: CPU fallback" >&2; exit 1; }

FAIL=0
for CFG in $CONFIGS; do
  NAME=$(python -c "import json;print(json.load(open('$CFG'))['name'])")
  OUT=$(python -c "import json;print(json.load(open('$CFG'))['output_dir'])")
  if [[ -f "$OUT/summary.json" ]]; then echo "--- $NAME already complete, skipping ---"; continue; fi
  echo "--- evaluating $NAME ---"
  # point the config's answerer at our port
  TMP="$OUT/_job_config_${SLURM_JOB_ID}.json"
  mkdir -p "$OUT"
  python -c "
import json
c = json.load(open('$CFG')); c.setdefault('answerer',{})['endpoint']='http://127.0.0.1:${PORT}'
json.dump(c, open('$TMP','w'), indent=1)"
  if ! python -m vidcurator.run evaluate --config "$TMP" 2>&1 | tee "logs/eval_${NAME}_${SLURM_JOB_ID}.log" | tail -5; then
    echo "ERROR: $NAME failed" >&2; FAIL=1
  fi
done
echo "=== [$SLURM_JOB_ID] pilot-multi finished (fail=$FAIL) ==="
exit "$FAIL"
