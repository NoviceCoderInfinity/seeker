#!/usr/bin/env bash
# Full wind-up: chain MLVU + final paper refresh after all learned GRPO jobs.
# Safe to re-run; skips finished steps.
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator

submit_a40() {
  sbatch --partition=a40 --qos=a40 "$@" 2>/tmp/vc_wind.err | awk '{print $NF}' || { cat /tmp/vc_wind.err; echo ""; }
}

# Find last job in the learned chain still queued/running (for dependency tail)
LAST=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk '
  /vc-lgrpo|vc-learned-grpo/ { print $1 }
' | tail -1)

if [[ -z "$LAST" ]]; then
  # fallback: last completed learned job id from sacct
  LAST=$(sacct -u "$USER" --starttime=2026-07-01 --format=JobID,JobName,State -P -n 2>/dev/null | \
    awk -F'|' '$2 ~ /vc-lgrpo|vc-learned-grpo/ && $3=="COMPLETED" && $1 !~ /\./ { print $1 }' | tail -1)
fi

echo "=== wind-up $(date) last_in_queue=${LAST:-none} ==="
squeue -u "$USER" 2>/dev/null || true

# MLVU (only missing benchmark)
if [[ ! -f runs/gpu_mlvu_qwen2_5_learned_grpo/summary.json ]]; then
  if ! squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -q vc-lgrpo-mlvu; then
    dep=""
    [[ -n "$LAST" ]] && dep="afterok:$LAST"
    args=(--job-name=vc-lgrpo-mlvu)
    [[ -n "$dep" ]] && args+=(--dependency="$dep")
    args+=(--export=ALL,CONFIG_PATH=configs/gpu_mlvu_qwen2_5_learned_grpo.json,RUN_DIR=runs/gpu_mlvu_qwen2_5_learned_grpo,PORT=8045,QA_PATH=data/processed/mlvu_val_150.jsonl,FRAMES_PATH=data/indices/mlvu_150_siglip_frames.jsonl)
    MLVU=$(submit_a40 "${args[@]}" jobs/26_learned_grpo_eval.sh)
    if [[ "$MLVU" =~ ^[0-9]+$ ]]; then
      echo "Queued MLVU -> $MLVU (dep=${dep:-none})"
      LAST="$MLVU"
    else
      echo "MLVU queue failed (QOS limit?) — retry when a slot opens"
    fi
  else
    echo "MLVU already queued"
  fi
else
  echo "MLVU done"
fi

# Cancel partial refresh; schedule final after tail + patch
scancel 144512 2>/dev/null || true
PATCH=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk '/ngqa-patch/ {print $1}')
DEPS=()
[[ -n "$PATCH" ]] && DEPS+=("afterok:$PATCH")
[[ -n "$LAST" ]] && DEPS+=("afterok:$LAST")

if [[ ${#DEPS[@]} -gt 0 ]]; then
  dep=$(IFS=:; echo "${DEPS[*]}")
  # SLURM wants afterok:J1:afterok:J2
  dep="${dep// /:afterok:}"
  REF=$(sbatch --partition=l40 --qos=l40 --dependency="$dep" \
    --job-name=vc-paper-final jobs/20_paper_refresh.sh 2>/tmp/vc_ref.err | awk '{print $NF}' || true)
  if [[ "$REF" =~ ^[0-9]+$ ]]; then
    echo "Final paper refresh -> $REF (dep=$dep)"
  else
    echo "Refresh queue failed: $(cat /tmp/vc_ref.err 2>/dev/null)"
  fi
fi

echo
squeue -u "$USER" 2>/dev/null || true
