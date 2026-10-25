#!/bin/bash
#SBATCH --output=logs/l4all_%j.out
#SBATCH --error=logs/l4all_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# L4 ledger-portability: replay Qwen-curated mlvu-full dde@8 buffers (frames + ledger)
# through all three other families sequentially in ONE job (submit-limit friendly).
set -uo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs reports

export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLM_MAX_PIXELS="${VLM_MAX_PIXELS:-200704}"
export PYTHONUNBUFFERED=1

PREDS=runs/full_v3_mlvu/predictions.jsonl
QA=data/processed/mlvu_val.jsonl
FRAMES=data/indices/mlvu_siglip_frames.jsonl

run_family() {  # model_path port out_md server_py
  local MODEL="$1" PORT="$2" OUT="$3" SPY="${4:-python}"
  if [[ -f "$OUT" ]]; then echo "--- $OUT exists, skipping ---"; return 0; fi
  echo "=== serving $MODEL on $PORT ==="
  "$SPY" scripts/serve_transformers_vlm.py --model "$MODEL" --port "$PORT" --dtype bfloat16 \
    > "logs/vlm_${SLURM_JOB_ID}_l4_$(basename "$MODEL").log" 2>&1 &
  local PID=$!
  local ready=0
  for attempt in $(seq 1 180); do
    if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5)" 2>/dev/null; then
      ready=1; break
    fi
    sleep 5
  done
  if [[ "$ready" -ne 1 ]]; then echo "ERROR: $MODEL server failed" >&2; kill "$PID" 2>/dev/null; return 1; fi
  python scripts/eval_cross_model.py --preds "$PREDS" --qa "$QA" --frames "$FRAMES" \
    --endpoint "http://127.0.0.1:${PORT}" --policies dde_curator uniform --budgets 8 \
    --with-ledger --out "$OUT" 2>&1 | tail -20
  local RC=$?
  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
  sleep 10
  return "$RC"
}

FAIL=0
run_family models/internvl3_5_8b 8545 reports/l4_ledger_port_mlvu_iv.md || FAIL=1
run_family models/llava_onevision_qwen2_7b_ov 8546 reports/l4_ledger_port_mlvu_ov.md || FAIL=1
run_family models/gemma_3_12b_it 8547 reports/l4_ledger_port_mlvu_g3.md "$HOME/envs/vlmserve/bin/python" || FAIL=1
echo "=== [$SLURM_JOB_ID] L4 all-family replay done (fail=$FAIL) ==="
exit "$FAIL"
