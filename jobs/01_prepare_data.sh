#!/bin/bash
#SBATCH --job-name=vidcurator-data-prep
#SBATCH --output=logs/data_prep_%j.out
#SBATCH --error=logs/data_prep_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/raw data/processed data/indices data/videos data/frames models runs

if [[ -d "$HOME/miniconda3/envs/vidcurator" ]]; then
    export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
elif [[ -f .conda-env/bin/activate ]]; then
    source .conda-env/bin/activate
elif [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
else
    echo "ERROR: no Python environment found (.conda-env or .venv)" >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
mkdir -p "$HF_HOME"

echo "=== [$SLURM_JOB_ID] data prep started at $(date) ==="
echo "HF_TOKEN set: $([[ -n ${HF_TOKEN:-} ]] && echo yes || echo no)"
echo "Python: $(which python)"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN not set; gated repos (LongVideoBench, MLVU) will be skipped" >&2
fi

echo "--- downloading non-gated dataset metadata ---"
python scripts/download_priority_assets.py --datasets --metadata-only --skip-existing || true

if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "--- downloading gated dataset metadata (LongVideoBench, MLVU) ---"
    python -c "
from huggingface_hub import snapshot_download
import os
token = os.environ['HF_TOKEN']
snapshot_download(repo_id='longvideobench/LongVideoBench', repo_type='dataset',
    local_dir='data/raw/longvideobench', token=token,
    allow_patterns=['lvb_val.json','validation-00000-of-00001.parquet','README.md'])
print('LVB metadata done')
snapshot_download(repo_id='MLVU/MLVU_Test', repo_type='dataset',
    local_dir='data/raw/mlvu', token=token,
    allow_patterns=['MLVU_Test/test_question.json','test-ground-truth/*.json','test_multi_choice_tasks.json','test_generation_tasks.json','README.md'])
print('MLVU metadata done')
" || echo "WARNING: gated download failed; check HF_TOKEN permissions" >&2
fi

echo "--- downloading model configs/tokenizers ---"
python scripts/download_priority_assets.py --models --metadata-only --skip-existing || true

echo "--- normalizing datasets ---"
python scripts/normalize_nextgqa.py \
    --csv data/raw/nextgqa/datasets/nextgqa/val.csv \
    --grounding-json data/raw/nextgqa/datasets/nextgqa/gsub_val.json \
    --frame2time-json data/raw/nextgqa/datasets/nextgqa/frame2time_val.json \
    --output-qa data/processed/nextgqa_val.jsonl \
    --output-frames data/indices/nextgqa_timestamp_frames.jsonl \
    --split val

python scripts/normalize_videomme.py
python scripts/normalize_nextqa.py
python scripts/normalize_egoschema.py

if [[ -f data/raw/longvideobench/lvb_val.json ]]; then
    python -m vidcurator.run normalize \
        --dataset longvideobench \
        --input data/raw/longvideobench/lvb_val.json \
        --output data/processed/longvideobench_val.jsonl
fi

if [[ -f data/raw/mlvu/test-ground-truth/test_mcq_gt.json ]]; then
    python -m vidcurator.run normalize \
        --dataset mlvu \
        --input data/raw/mlvu/test-ground-truth/test_mcq_gt.json \
        --output data/processed/mlvu_val.jsonl
fi

echo "--- creating 150-example kill splits ---"
for ds in nextgqa_val videomme_long egoschema_val nextqa_temporal_val longvideobench_val mlvu_val; do
    if [[ -f data/processed/${ds}.jsonl ]]; then
        python scripts/make_split.py \
            --input data/processed/${ds}.jsonl \
            --output data/processed/${ds}_150.jsonl \
            --limit 150 --seed 13
    fi
done

echo "--- readiness check ---"
python scripts/check_research_readiness.py

echo "=== [$SLURM_JOB_ID] data prep finished at $(date) ==="
