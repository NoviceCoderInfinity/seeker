#!/bin/bash
# Full-scale (full-val) fan-out launcher for the parallel-node run — see plan5.md §3/§5 + HANDOFF_FULLSCALE.md.
# PORTABLE: resolves the repo root from its own location, so it works from ANY user's copy
# (do NOT hardcode /home/vis-comp/22b3929/...). Submits one jobs/12 per dataset for a given WAVE + MODEL,
# writing fs_-prefixed run-dirs so nothing collides with the primary user's r1_v2_/r2_v2_ runs.
#
# Usage (run from the SECOND user's copy, after configs are generated with gen_fullscale_configs.py):
#   WAVE=headline MODEL=qwen2_5_vl_7b bash jobs/50_fullscale_submit.sh
#   WAVE=selectors MODEL=qwen2_5_vl_7b bash jobs/50_fullscale_submit.sh
#   WAVE=ablations MODEL=qwen2_5_vl_7b bash jobs/50_fullscale_submit.sh
#   WAVE=headline  MODEL=internvl3_5_8b bash jobs/50_fullscale_submit.sh   # cross-family (plan5 §5)
# Optional env:
#   DATASETS="nextgqa,videomme"   restrict the fan-out (default = all 6)
#   START_PORT=8100               first port (each ds gets START_PORT+i)
#   EXCLUDE="cn11-dgx,cn14-dgx,cn15-dgx,cn17-dgx"   nodes to avoid (verify with `sinfo -R` first!)
#   DRYRUN=1                      print the sbatch lines without submitting
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs runs

WAVE="${WAVE:?set WAVE=headline|selectors|ablations|bsf}"
MODEL="${MODEL:-qwen2_5_vl_7b}"
START_PORT="${START_PORT:-8100}"
EXCLUDE="${EXCLUDE:-cn11-dgx,cn14-dgx,cn15-dgx,cn17-dgx}"
DRYRUN="${DRYRUN:-0}"

# dataset -> full-val QA | full SigLIP index | partition (heavy/long videos -> dgx, light -> l40)
# Adjust partition names to THIS cluster if they differ (verify with `sinfo`).
JOBS=(
  "nextgqa|data/processed/nextgqa_val.jsonl|data/indices/nextgqa_siglip_frames.jsonl|dgx"
  "nextqa|data/processed/nextqa_temporal_val.jsonl|data/indices/nextqa_temporal_siglip_frames.jsonl|dgx"
  "videomme|data/processed/videomme_long.jsonl|data/indices/videomme_long_siglip_frames.jsonl|dgx"
  "egoschema|data/processed/egoschema_val.jsonl|data/indices/egoschema_siglip_frames.jsonl|l40"
  "mlvu|data/processed/mlvu_val.jsonl|data/indices/mlvu_siglip_frames.jsonl|l40"
  "longvideobench|data/processed/longvideobench_val.jsonl|data/indices/longvideobench_siglip_frames.jsonl|dgx"
)

# optional dataset filter
declare -A WANT=()
if [[ -n "${DATASETS:-}" ]]; then IFS=',' read -ra _d <<< "$DATASETS"; for d in "${_d[@]}"; do WANT[$d]=1; done; fi

i=0
for entry in "${JOBS[@]}"; do
  IFS='|' read -r name qa frames part <<< "$entry"
  [[ -n "${DATASETS:-}" && -z "${WANT[$name]:-}" ]] && continue
  port=$(( START_PORT + i )); i=$(( i + 1 ))
  cfg="configs/fs_${name}_${MODEL}_${WAVE}.json"
  run="runs/fs_${name}_${MODEL}_${WAVE}"
  if [[ ! -f "$cfg" ]]; then
    echo "!! missing $cfg — generate it first: python scripts/gen_fullscale_configs.py --wave $WAVE --model $MODEL" >&2
    continue
  fi
  if [[ -f "$run/summary.json" ]]; then
    echo "== skip $name/$WAVE — $run/summary.json already complete"; continue
  fi
  CMD=(sbatch --partition="$part" --qos="$part" --gres=gpu:1 --exclude="$EXCLUDE"
       --job-name="vc-fs-${WAVE}-${MODEL}-${name}"
       --export=ALL,CONFIG_PATH="$cfg",MODEL_PATH="models/${MODEL}",QA_PATH="$qa",FRAMES_PATH="$frames",RUN_DIR="$run",PORT="$port",JOB_LABEL="fs-${WAVE}-${name}"
       jobs/12_zero_shot_single.sh)
  echo "-> $name  part=$part port=$port  ($cfg)"
  if [[ "$DRYRUN" == "1" ]]; then printf '   %q ' "${CMD[@]}"; echo; else "${CMD[@]}"; fi
done
echo "fullscale WAVE=$WAVE MODEL=$MODEL dispatched (check: squeue -u \$USER)."
