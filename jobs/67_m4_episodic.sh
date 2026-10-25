#!/bin/bash
#SBATCH --output=logs/m4ep_%j.out
#SBATCH --error=logs/m4ep_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# M4 gate: episodic cross-question memory (video-level Q&A cache), training-free.
# Arms: control (ledger only), episodic-pred (realized cache), episodic-gold (ceiling).
set -uo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs reports
export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLM_MAX_PIXELS="${VLM_MAX_PIXELS:-200704}"
export PYTHONUNBUFFERED=1

PORT=8564
python scripts/serve_transformers_vlm.py --model models/qwen2_5_vl_7b --port "$PORT" --dtype bfloat16 \
  > "logs/vlm_${SLURM_JOB_ID}_m4ep.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null' EXIT
for attempt in $(seq 1 180); do
  python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5)" 2>/dev/null && break
  sleep 5
done

run_arm() {  # preds qa frames budget out extra...
  local PREDS="$1" QA="$2" FRAMES="$3" BUD="$4" OUT="$5"; shift 5
  [[ -f "$OUT" ]] && { echo "--- $OUT exists, skipping ---"; return 0; }
  python scripts/eval_cross_model.py --preds "$PREDS" --qa "$QA" --frames "$FRAMES" \
    --endpoint "http://127.0.0.1:${PORT}" --policies dde_curator --budgets "$BUD" \
    --with-ledger "$@" --out "$OUT" 2>&1 | tail -6
}

FAIL=0
LP=runs/dde_lvbench_heldout/predictions.jsonl; LQ=data/processed/heldout/lvbench_heldout.jsonl; LF=data/indices/lvbench_siglip_frames.jsonl
MP=runs/full_v3_mlvu/predictions.jsonl;        MQ=data/processed/mlvu_val.jsonl;               MF=data/indices/mlvu_siglip_frames.jsonl
run_arm "$LP" "$LQ" "$LF" 8 reports/m4_lvb_control.md                    || FAIL=1
run_arm "$LP" "$LQ" "$LF" 8 reports/m4_lvb_ep_pred.md --episodic pred    || FAIL=1
run_arm "$LP" "$LQ" "$LF" 8 reports/m4_lvb_ep_gold.md --episodic gold    || FAIL=1
run_arm "$MP" "$MQ" "$MF" 8 reports/m4_mlvu_ep_pred.md --episodic pred   || FAIL=1
run_arm "$MP" "$MQ" "$MF" 8 reports/m4_mlvu_ep_gold.md --episodic gold   || FAIL=1
echo "=== [$SLURM_JOB_ID] M4 episodic done (fail=$FAIL) ==="
exit "$FAIL"
