#!/usr/bin/env bash
# Final wind-up: learned GRPO eval + artifact export + paper refresh chain.
set -euo pipefail
ROOT="/home/vis-comp/22b3929/vidcurator"
cd "$ROOT"

submit_a40() {
  local label="$1"; shift
  if sbatch --partition=a40 --qos=a40 "$@" >/tmp/vc_windup.txt 2>&1; then
    echo "  QUEUED $label -> $(tail -1 /tmp/vc_windup.txt)"
    tail -1 /tmp/vc_windup.txt | awk '{print $NF}'
  else
    echo "  FAILED $label: $(cat /tmp/vc_windup.txt)"
    echo ""
  fi
}

submit_l40() {
  local label="$1"; shift
  if sbatch --partition=l40 --qos=l40 "$@" >/tmp/vc_windup.txt 2>&1; then
    echo "  QUEUED $label -> $(tail -1 /tmp/vc_windup.txt)"
    tail -1 /tmp/vc_windup.txt | awk '{print $NF}'
  else
    echo "  FAILED $label: $(cat /tmp/vc_windup.txt)"
    echo ""
  fi
}

echo "=== wind-up chain $(date) ==="
squeue -u "$USER" 2>/dev/null || true
echo

# Cancel stuck learned eval if still waiting on dead VLM
if squeue -h -j 144482 &>/dev/null; then
  out=$(tail -1 logs/learned_grpo_eval_144482.out 2>/dev/null || echo "")
  if [[ "$out" == *"Waiting for VLM"* ]] && [[ ! -s logs/serve_learned_grpo_144482.log ]]; then
    echo "Cancelling stuck job 144482 (VLM never started)"
    scancel 144482 2>/dev/null || true
    sleep 2
  fi
fi

LEARNED_JID=""
if [[ ! -f runs/gpu_nextgqa_qwen2_5_learned_grpo_analysis/summary.json ]]; then
  if ! squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -q vc-learned-grpo; then
    echo "[1] Learned GRPO downstream eval (NExT-GQA analysis n=150)..."
    LEARNED_JID=$(submit_a40 "learned-grpo-eval" jobs/26_learned_grpo_eval.sh)
  else
    echo "[1] SKIP learned eval (already queued/running)"
  fi
else
  echo "[1] SKIP learned eval (summary.json exists)"
fi

echo "[2] Artifact export (all completed runs)..."
EXPORT_JID=$(submit_l40 "artifact-export" jobs/05_export_artifacts.sh)

if [[ -n "$LEARNED_JID" ]]; then
  echo "[3] Paper refresh after learned eval ($LEARNED_JID)..."
  submit_a40 "paper-refresh-final" --dependency=afterok:"$LEARNED_JID" jobs/20_paper_refresh.sh
else
  echo "[3] Paper refresh now (learned eval already done or running separately)..."
  submit_l40 "paper-refresh-now" jobs/20_paper_refresh.sh
fi

echo
echo "=== final queue ==="
squeue -u "$USER" 2>/dev/null || true
