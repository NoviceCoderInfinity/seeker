#!/usr/bin/env bash
# Submit MLVU learned eval + final paper refresh (call after queue has a slot).
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator

LAST="${1:-144507}"
RUN=runs/gpu_mlvu_qwen2_5_learned_grpo

if [[ -f "$RUN/summary.json" ]]; then
  echo "MLVU already done"
  exit 0
fi

MLVU=$(sbatch --partition=a40 --qos=a40 --dependency=afterok:"$LAST" --job-name=vc-lgrpo-mlvu \
  --export=ALL,CONFIG_PATH=configs/gpu_mlvu_qwen2_5_learned_grpo.json,RUN_DIR="$RUN",PORT=8045,QA_PATH=data/processed/mlvu_val_150.jsonl,FRAMES_PATH=data/indices/mlvu_150_siglip_frames.jsonl \
  jobs/26_learned_grpo_eval.sh | awk '{print $NF}')
echo "MLVU job $MLVU (after $LAST)"

scancel 144510 2>/dev/null || true
sbatch --partition=l40 --qos=l40 \
  --dependency=afterok:144509:afterok:"$MLVU" \
  --job-name=vc-paper-final jobs/20_paper_refresh.sh
