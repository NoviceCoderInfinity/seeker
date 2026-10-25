#!/bin/bash
#SBATCH --job-name=vidcurator-zs-single
#SBATCH --output=logs/zs_single_%x_%j.out
#SBATCH --error=logs/zs_single_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Single-dataset zero-shot: serve one VLM + evaluate one config + export artifacts.
# Required env: CONFIG_PATH, MODEL_PATH, QA_PATH, FRAMES_PATH, RUN_DIR, PORT
# Optional: JOB_LABEL (for log naming)
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs runs

export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
# Avoid fragmentation OOM: lets the allocator reclaim reserved-but-unallocated blocks
# (the R6-mlvu InternVL crash had 12GB reserved-unusable while asking for 6GB more).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Per-frame visual-token cap. Default 200704 px (=256 tokens / ~448x448) keeps
# K=16/32 frame contexts within budget. Override by exporting VLM_MAX_PIXELS.
export VLM_MAX_PIXELS="${VLM_MAX_PIXELS:-200704}"
export VLM_MIN_PIXELS="${VLM_MIN_PIXELS:-3136}"

PY="$HOME/miniconda3/envs/vidcurator/bin/python"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH required}"
QA_PATH="${QA_PATH:?QA_PATH required}"
FRAMES_PATH="${FRAMES_PATH:?FRAMES_PATH required}"
RUN_DIR="${RUN_DIR:?RUN_DIR required}"
PORT="${PORT:-8000}"
LABEL="${JOB_LABEL:-$(basename "$CONFIG_PATH" .json)}"

echo "=== [$SLURM_JOB_ID] zero-shot single: $LABEL at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — check torch build (need cu124)'; print('cuda OK:', torch.cuda.get_device_name(0))"

if [[ -f "$RUN_DIR/summary.json" ]]; then
    echo "Already complete: $RUN_DIR/summary.json — skipping"
    exit 0
fi

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "--- Serving $MODEL_PATH on port $PORT ---"
CUDA_VISIBLE_DEVICES=0 \
    $PY scripts/serve_transformers_vlm.py \
    --model "$MODEL_PATH" --port "$PORT" --dtype bfloat16 \
    --max-pixels "$VLM_MAX_PIXELS" --min-pixels "$VLM_MIN_PIXELS" \
    > "logs/vlm_${SLURM_JOB_ID}_${LABEL}.log" 2>&1 &
SERVER_PID=$!

echo "Waiting for VLM..."
ready=0
for attempt in $(seq 1 180); do
    if $PY -c "
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
    tail -30 "logs/vlm_${SLURM_JOB_ID}_${LABEL}.log" >&2
    exit 1
fi

# Verify model is on GPU (not CPU)
if grep -q "Model loaded on cpu" "logs/vlm_${SLURM_JOB_ID}_${LABEL}.log"; then
    echo "ERROR: model loaded on CPU — aborting" >&2
    exit 1
fi

echo "--- Evaluating $CONFIG_PATH (endpoint localhost:$PORT) ---"
mkdir -p "$RUN_DIR"
JOB_CONFIG="$RUN_DIR/_job_config_${SLURM_JOB_ID}.json"
$PY - <<PY
import json
from pathlib import Path
src = Path("$CONFIG_PATH")
cfg = json.loads(src.read_text())
cfg["output_dir"] = "$RUN_DIR"
cfg.setdefault("answerer", {})["endpoint"] = "http://localhost:${PORT}"
Path("$JOB_CONFIG").write_text(json.dumps(cfg, indent=2))
PY
$PY -m vidcurator.run evaluate --config "$JOB_CONFIG" \
    2>&1 | tee "logs/eval_${LABEL}_${SLURM_JOB_ID}.log"

echo "--- Exporting artifacts ---"
$PY scripts/export_paper_artifacts.py --run-dir "$RUN_DIR" || true
$PY scripts/export_qualitative_traces.py \
    --run-dir "$RUN_DIR" \
    --qa-path "$QA_PATH" \
    --frames-path "$FRAMES_PATH" \
    --policy scripted_curator --max-examples 10 || true

echo "--- Results ---"
$PY -c "
import json
s = json.load(open('$RUN_DIR/summary.json'))
for k in sorted(s['groups']):
    if 'scripted_curator@8' in k or 'uniform@8' in k or 'uniform@16' in k:
        g = s['groups'][k]
        print(f\"  {k}: acc={g['accuracy']:.3f} frames={g['avg_frames']:.1f} fep={g.get('fep','n/a')}\")
if s.get('retrieval_recall_at_k'):
    print('  recall@k:', s['retrieval_recall_at_k'])
"

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
trap - EXIT

echo "=== [$SLURM_JOB_ID] finished $LABEL at $(date) ==="
