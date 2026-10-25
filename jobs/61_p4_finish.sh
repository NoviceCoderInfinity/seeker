#!/bin/bash
# P4 finish chain: merge the p4_multi GRPO adapter, then run learned-curator heldout
# evals on all four datasets with one served answerer. Submit with afterok:<grpo job>.
# Env: GRPO_RUN_DIR, MERGED_DIR, CONFIGS (space-separated), PORT, STEP (optional).
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs runs

export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export ACTION_POLICY_DEVICE="${ACTION_POLICY_DEVICE:-cuda:1}"
export VLM_MAX_PIXELS="${VLM_MAX_PIXELS:-200704}"

VID_PY="$HOME/miniconda3/envs/vidcurator/bin/python"
TRAIN_PY="${TRAIN_PY:-$HOME/miniconda3/envs/vidcurator-train/bin/python}"  # has peft; needed by action_policy adapter load
GRPO_PY="${GRPO_PY:-$HOME/miniconda3/envs/vidcurator-grpo/bin/python}"
GRPO_RUN_DIR="${GRPO_RUN_DIR:-runs/grpo/p4_multi}"
MERGED_DIR="${MERGED_DIR:-runs/grpo/p4_multi_lora}"
PORT="${PORT:-8400}"
: "${CONFIGS:?CONFIGS env var required}"

# --- 1. merge (skip if already merged) ---
if [[ ! -f "$MERGED_DIR/lora_adapter/adapter_config.json" ]]; then
  STEP="${STEP:-$(tr -d '[:space:]' < "$GRPO_RUN_DIR/latest_checkpointed_iteration.txt")}"
  ACTOR_DIR="$GRPO_RUN_DIR/global_step_${STEP}/actor"
  [[ -d "$ACTOR_DIR" ]] || { echo "ERROR: missing $ACTOR_DIR" >&2; exit 1; }
  echo "=== [$SLURM_JOB_ID] merging $ACTOR_DIR -> $MERGED_DIR ==="
  rm -rf "$MERGED_DIR"
  "$GRPO_PY" -m verl.model_merger merge --backend fsdp \
    --local_dir "$ACTOR_DIR" --target_dir "$MERGED_DIR"
  [[ -f "$MERGED_DIR/lora_adapter/adapter_config.json" ]] || { echo "ERROR: merge failed" >&2; exit 1; }
fi
echo "=== adapter ready: $MERGED_DIR/lora_adapter ==="

# --- 2. serve answerer on GPU0 ---
cleanup() { [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT
CUDA_VISIBLE_DEVICES=0 "$VID_PY" scripts/serve_transformers_vlm.py \
  --model models/qwen2_5_vl_7b --port "$PORT" --dtype bfloat16 \
  --max-pixels "$VLM_MAX_PIXELS" > "logs/vlm_${SLURM_JOB_ID}_p4finish.log" 2>&1 &
SERVER_PID=$!
ready=0
for attempt in $(seq 1 180); do
  if "$VID_PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5)" 2>/dev/null; then
    ready=1; echo "VLM ready (${attempt})"; break
  fi
  sleep 5
done
[[ "$ready" -eq 1 ]] || { echo "ERROR: VLM failed" >&2; tail -30 "logs/vlm_${SLURM_JOB_ID}_p4finish.log" >&2; exit 1; }

# --- 3. loop learned evals (action policy on cuda:1) ---
FAIL=0
for CFG in $CONFIGS; do
  NAME=$("$VID_PY" -c "import json;print(json.load(open('$CFG'))['name'])")
  OUT=$("$VID_PY" -c "import json;print(json.load(open('$CFG'))['output_dir'])")
  if [[ -f "$OUT/summary.json" ]]; then echo "--- $NAME already complete ---"; continue; fi
  echo "--- evaluating $NAME ---"
  mkdir -p "$OUT"
  CUDA_VISIBLE_DEVICES=0,1 "$TRAIN_PY" -m vidcurator.run evaluate --config "$CFG" \
    2>&1 | tee "logs/eval_${NAME}_${SLURM_JOB_ID}.log" || FAIL=1
done
echo "=== [$SLURM_JOB_ID] p4 finish done (FAIL=$FAIL) ==="
exit $FAIL
