#!/bin/bash
#SBATCH --output=logs/l1l3_%j.out
#SBATCH --error=logs/l1l3_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# L1 (ledger-only: text memory, zero images) + L3 (timestamp-strip: does the answerer
# use 'when'?) — replay stored dde@8 buffers through the ORIGINAL Qwen answerer.
# Control arm = same replay with untouched ledger, so all three arms share serving
# nondeterminism and pair per-question.
set -uo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs reports

export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLM_MAX_PIXELS="${VLM_MAX_PIXELS:-200704}"
export PYTHONUNBUFFERED=1

PORT=8560
python scripts/serve_transformers_vlm.py --model models/qwen2_5_vl_7b --port "$PORT" --dtype bfloat16 \
  > "logs/vlm_${SLURM_JOB_ID}_l1l3.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null' EXIT
ready=0
for attempt in $(seq 1 180); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5)" 2>/dev/null; then
    ready=1; break
  fi
  sleep 5
done
[[ "$ready" -eq 1 ]] || { echo "ERROR: VLM failed to start" >&2; exit 1; }

run_arm() {  # preds qa frames out extra_flags...
  local PREDS="$1" QA="$2" FRAMES="$3" OUT="$4"; shift 4
  if [[ -f "$OUT" ]]; then echo "--- $OUT exists, skipping ---"; return 0; fi
  python scripts/eval_cross_model.py --preds "$PREDS" --qa "$QA" --frames "$FRAMES" \
    --endpoint "http://127.0.0.1:${PORT}" --policies dde_curator --budgets 8 \
    --with-ledger "$@" --out "$OUT" 2>&1 | tail -8
}

FAIL=0
MP=runs/full_v3_mlvu/predictions.jsonl;         MQ=data/processed/mlvu_val.jsonl;          MF=data/indices/mlvu_siglip_frames.jsonl
NP=runs/full_v3_nextgqa_half/predictions.jsonl; NQ=data/processed/nextgqa_val_half.jsonl;  NF=data/indices/nextgqa_siglip_frames.jsonl
run_arm "$MP" "$MQ" "$MF" reports/l1l3_mlvu_control.md                     || FAIL=1
run_arm "$MP" "$MQ" "$MF" reports/l1l3_mlvu_striptime.md --strip-timestamps || FAIL=1
run_arm "$MP" "$MQ" "$MF" reports/l1l3_mlvu_ledgeronly.md --ledger-only     || FAIL=1
run_arm "$NP" "$NQ" "$NF" reports/l1l3_nextgqa_control.md                     || FAIL=1
run_arm "$NP" "$NQ" "$NF" reports/l1l3_nextgqa_striptime.md --strip-timestamps || FAIL=1
run_arm "$NP" "$NQ" "$NF" reports/l1l3_nextgqa_ledgeronly.md --ledger-only     || FAIL=1
echo "=== [$SLURM_JOB_ID] L1/L3 done (fail=$FAIL) ==="
exit "$FAIL"
