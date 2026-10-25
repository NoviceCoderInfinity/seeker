#!/usr/bin/env bash
# Submit remaining vidcurator GPU jobs with dependencies (no smoke).
set -euo pipefail
cd "$HOME/vidcurator"
mkdir -p logs

echo "=== Submitting remaining vidcurator jobs ==="

# Qwen3 multi-dataset (DGX) — main gap
if [[ ! -f runs/gpu_videomme_qwen3_zero_shot/summary.json ]]; then
  QWEN3=$(sbatch --parsable jobs/11_zero_shot_qwen3_multi_dgx.sh)
  echo "Qwen3 multi (DGX): job $QWEN3"
  ART=$(sbatch --parsable --dependency=afterok:$QWEN3 jobs/05_export_artifacts.sh)
  echo "Artifacts (after Qwen3): job $ART"
else
  echo "Qwen3 multi already done"
  ART=$(sbatch --parsable jobs/05_export_artifacts.sh)
  echo "Artifacts: job $ART"
fi

echo
echo "Monitor: squeue -u \$USER"
echo "Logs:    logs/zs_qwen3_multi_<id>.out"
echo "Watch:   tail -f logs/pipeline_watch.log"
