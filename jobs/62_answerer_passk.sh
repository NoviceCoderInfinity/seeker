#!/bin/bash
# Answerer-side pass@k probe: serve the answerer (sampling-patched server), then replay
# stored DdE buffers k times at temperature via scripts/answerer_passk.py.
# Env: PREDS (shard to replay), QA, FRAMES, PORT, K, TEMP, OUT.
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs runs

export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export VLM_MAX_PIXELS="${VLM_MAX_PIXELS:-200704}"

VID_PY="$HOME/miniconda3/envs/vidcurator/bin/python"
PORT="${PORT:-8485}"
PREDS="${PREDS:-runs/p0_v3control_nextgqa/_shards/b8_dde_curator.jsonl}"
QA="${QA:-data/processed/heldout/nextgqa_heldout_p120.jsonl}"
FRAMES="${FRAMES:-data/indices/nextgqa_siglip_frames.jsonl}"
K="${K:-32}"
TEMP="${TEMP:-1.0}"
OUT="${OUT:-runs/answerer_passk32_nextgqa_b8/votes.jsonl}"

cleanup() { [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT
CUDA_VISIBLE_DEVICES=0 "$VID_PY" scripts/serve_transformers_vlm.py \
  --model models/qwen2_5_vl_7b --port "$PORT" --dtype bfloat16 \
  --max-pixels "$VLM_MAX_PIXELS" > "logs/vlm_${SLURM_JOB_ID}_apassk.log" 2>&1 &
SERVER_PID=$!
ready=0
for attempt in $(seq 1 180); do
  if "$VID_PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5)" 2>/dev/null; then
    ready=1; echo "VLM ready (${attempt})"; break
  fi
  sleep 5
done
[[ "$ready" -eq 1 ]] || { echo "ERROR: VLM failed" >&2; tail -30 "logs/vlm_${SLURM_JOB_ID}_apassk.log" >&2; exit 1; }

"$VID_PY" scripts/answerer_passk.py --preds "$PREDS" --qa "$QA" --frames "$FRAMES" \
  --endpoint "http://127.0.0.1:${PORT}" --k "$K" --temperature "$TEMP" --out "$OUT" \
  2>&1 | tee "logs/answerer_passk_${SLURM_JOB_ID}.log"
echo "=== [$SLURM_JOB_ID] answerer passk done ==="
