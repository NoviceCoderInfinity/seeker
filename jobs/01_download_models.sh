#!/bin/bash
#SBATCH --job-name=vidcurator-dl-models
#SBATCH --output=logs/dl_models_%j.out
#SBATCH --error=logs/dl_models_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs models

export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

PY="$HOME/miniconda3/envs/vidcurator/bin/python"

echo "=== [$SLURM_JOB_ID] model download started at $(date) ==="

download_model() {
    local name="$1"
    local dir="models/$name"
    if ls "$dir"/*.safetensors >/dev/null 2>&1; then
        echo "  $name: already present ($(ls "$dir"/*.safetensors | wc -l) shards)"
        return 0
    fi
    echo "  $name: downloading..."
    $PY scripts/download_hf_assets.py --kind model --name "$name" || {
        echo "WARN: download_hf_assets failed for $name, trying priority script..."
        $PY scripts/download_priority_assets.py --models --only "$name" || true
    }
}

# Priority models for backbone sweep
download_model internvl3_5_8b
download_model llava_video_7b_qwen2
download_model qwen3_vl_8b_thinking

echo "--- model inventory ---"
for d in models/*/; do
    n=$(ls "$d"*.safetensors 2>/dev/null | wc -l)
    sz=$(du -sh "$d" 2>/dev/null | cut -f1)
    echo "  $(basename $d): safetensors=$n size=$sz"
done

echo "=== [$SLURM_JOB_ID] model download finished at $(date) ==="
