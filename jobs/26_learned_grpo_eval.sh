#!/bin/bash
#SBATCH --job-name=vc-learned-grpo
#SBATCH --output=logs/learned_grpo_eval_%j.out
#SBATCH --error=logs/learned_grpo_eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Downstream eval: GRPO LoRA adapter drives learned_curator vs scripted_curator.
# Serves base VLM for QA answering; loads adapter in-process for action JSON.
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs runs

export HF_HOME="${HF_HOME:-/home/vis-comp/22b3929/vidcurator/.hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# GPU layout: served VLM on GPU0, action adapter on GPU1 (needs --gres=gpu:2).
# Override CUDA_VISIBLE_DEVICES=0 + ACTION_POLICY_DEVICE=cpu for a single-GPU fallback.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1
# Adapter on its own GPU so learned_curator runs at GPU speed (CPU took >12h and timed out).
export ACTION_POLICY_DEVICE="${ACTION_POLICY_DEVICE:-cuda:1}"

VID_PY="${VID_PY:-$HOME/miniconda3/envs/vidcurator/bin/python}"
TRAIN_PY="${TRAIN_PY:-$HOME/miniconda3/envs/vidcurator-train/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/gpu_nextgqa_qwen2_5_learned_grpo_analysis.json}"
MODEL_PATH="${MODEL_PATH:-models/qwen2_5_vl_7b}"
PORT="${PORT:-8040}"
RUN_DIR="${RUN_DIR:-runs/gpu_nextgqa_qwen2_5_learned_grpo_analysis}"
QA_PATH="${QA_PATH:-data/processed/nextgqa_val_150_siglip.jsonl}"
FRAMES_PATH="${FRAMES_PATH:-data/indices/nextgqa_siglip_frames.jsonl}"

echo "=== [$SLURM_JOB_ID] learned GRPO downstream eval at $(date) ==="
echo "Node: $(hostname)"
echo "Config: $CONFIG_PATH"

if [[ -f "$RUN_DIR/summary.json" ]]; then
  echo "Already complete: $RUN_DIR/summary.json — skipping"
  exit 0
fi

ADAPTER="${ADAPTER_PATH:-runs/grpo/qwen2_5_vl_7b_grpo_lora/lora_adapter}"
if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
  echo "ERROR: missing GRPO adapter at $ADAPTER" >&2
  exit 1
fi

$VID_PY -c "import torch; assert torch.cuda.is_available(); print('cuda OK:', torch.cuda.get_device_name(0))"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

SERVE_LOG="logs/serve_learned_grpo_${SLURM_JOB_ID}.log"
echo "--- Serving $MODEL_PATH on port $PORT (base VLM for answering) ---"
CUDA_VISIBLE_DEVICES=0 \
  $VID_PY scripts/serve_transformers_vlm.py \
    --model "$MODEL_PATH" --port "$PORT" --dtype bfloat16 \
    > "$SERVE_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for VLM..."
ready=0
for attempt in $(seq 1 180); do
  if $VID_PY -c "
import json, urllib.request
with urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5) as r:
    print(json.loads(r.read().decode()).get('data',[{}])[0].get('id','ok'))
" 2>/dev/null; then
    ready=1
    echo "VLM ready after $attempt attempts ($(( attempt * 5 ))s)"
    break
  fi
  sleep 5
done
if [[ "$ready" -ne 1 ]]; then
  echo "ERROR: VLM failed to start" >&2
  tail -40 "$SERVE_LOG" >&2 || true
  exit 1
fi
if grep -q "Model loaded on cpu" "$SERVE_LOG"; then
  echo "ERROR: model loaded on CPU — aborting" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
JOB_CONFIG="$RUN_DIR/_job_config_${SLURM_JOB_ID}.json"
$VID_PY - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("$CONFIG_PATH").read_text())
cfg["output_dir"] = "$RUN_DIR"
cfg.setdefault("answerer", {})["endpoint"] = "http://127.0.0.1:${PORT}"
# Inject the adapter under eval so ADAPTER_PATH actually reaches evaluate.py
# (previously only existence-checked, so every run silently used the base config's adapter).
cfg.setdefault("action_policy", {})["adapter_path"] = "$ADAPTER"
Path("$JOB_CONFIG").write_text(json.dumps(cfg, indent=2))
PY

echo "--- Evaluating $JOB_CONFIG (learned_curator + scripted_curator) ---"
$TRAIN_PY -m vidcurator.run evaluate --config "$JOB_CONFIG" \
  2>&1 | tee "logs/eval_learned_grpo_${SLURM_JOB_ID}.log"

echo "--- Export artifacts ---"
$VID_PY scripts/export_paper_artifacts.py --run-dir "$RUN_DIR" || echo "WARN: export partial"
$VID_PY scripts/export_qualitative_traces.py \
  --run-dir "$RUN_DIR" --qa-path "$QA_PATH" --frames-path "$FRAMES_PATH" \
  --policy learned_curator --max-examples 10 || true

echo "--- Headline ---"
$VID_PY -c "
import json
s = json.load(open('$RUN_DIR/summary.json'))
for k in sorted(s.get('groups', {})):
    if 'learned_curator@' in k or 'scripted_curator@8' in k:
        g = s['groups'][k]
        print(f\"  {k}: acc={g['accuracy']:.3f} frames={g['avg_frames']:.1f} fep={g.get('fep','n/a')}\")
"

echo "=== [$SLURM_JOB_ID] learned GRPO eval finished at $(date) ==="
