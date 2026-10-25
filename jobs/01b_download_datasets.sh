#!/bin/bash
#SBATCH --job-name=vidcurator-dl-data
#SBATCH --output=logs/dl_datasets_%j.out
#SBATCH --error=logs/dl_datasets_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs data/videos/longvideobench data/videos/mlvu data/raw

export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

PY="$HOME/miniconda3/envs/vidcurator/bin/python"

echo "=== [$SLURM_JOB_ID] dataset download started at $(date) ==="

# LongVideoBench + MLVU (missing videos)
for ds in longvideobench mlvu; do
    echo "--- downloading $ds ---"
    $PY scripts/download_priority_assets.py --datasets --only "$ds" --skip-existing --allow-large-video \
        2>&1 | tee -a "logs/download_${ds}_${SLURM_JOB_ID}.log" || echo "WARN: $ds download had issues"
done

echo "--- video inventory ---"
du -sh data/videos/* 2>/dev/null || true

echo "=== [$SLURM_JOB_ID] dataset download finished at $(date) ==="
