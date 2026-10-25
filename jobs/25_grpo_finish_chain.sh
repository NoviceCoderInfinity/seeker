#!/usr/bin/env bash
# Chain post-GRPO validation + paper refresh after the running full GRPO job.
set -euo pipefail
ROOT="/home/vis-comp/22b3929/vidcurator"
cd "$ROOT"

GRPO_JOB="${GRPO_JOB:-144064}"
SKIP_MERGE="${SKIP_MERGE:-1}"

if ! squeue -h -j "$GRPO_JOB" &>/dev/null; then
  state=$(sacct -j "$GRPO_JOB" --format=State -P -n 2>/dev/null | head -1 || echo "unknown")
  if [[ "$state" != "COMPLETED" ]]; then
    echo "WARN: GRPO job $GRPO_JOB state=$state (not running; will still queue validate if COMPLETED soon)"
  fi
fi

echo "=== GRPO finish chain (after $GRPO_JOB) $(date) ==="

VAL_JID=$(sbatch --partition=a40 --qos=a40 \
  --export=ALL,SKIP_MERGE="$SKIP_MERGE" \
  jobs/24_post_grpo_validate.sh | awk '{print $NF}')
echo "Queued post-GRPO validate -> job $VAL_JID (SKIP_MERGE=$SKIP_MERGE)"

REF_JID=$(sbatch --partition=a40 --qos=a40 \
  --dependency=afterok:"$VAL_JID" \
  jobs/20_paper_refresh.sh | awk '{print $NF}')
echo "Queued paper refresh after validate -> job $REF_JID"

squeue -u "$USER" 2>/dev/null || true
