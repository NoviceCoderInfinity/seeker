#!/bin/bash
#SBATCH --job-name=vidcurator-artifacts
#SBATCH --output=logs/artifacts_%j.out
#SBATCH --error=logs/artifacts_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

if [[ -d "$HOME/miniconda3/envs/vidcurator" ]]; then
    export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
elif [[ -f .conda-env/bin/activate ]]; then
    source .conda-env/bin/activate
elif [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
else
    echo "ERROR: no Python environment found" >&2
    exit 1
fi

echo "=== [$SLURM_JOB_ID] artifact export started at $(date) ==="

echo "--- exporting paper artifacts for all zero-shot runs ---"
for run_dir in runs/zero_shot/*/; do
    if [[ -f "${run_dir}summary.json" ]]; then
        echo "  exporting $(basename "$run_dir")"
        python scripts/export_paper_artifacts.py --run-dir "${run_dir%/}" || true
    fi
done

echo "--- exporting artifacts for standalone runs ---"
for run_dir in runs/kill_test_fixture runs/gpu_*/; do
    if [[ -f "${run_dir}summary.json" ]]; then
        echo "  exporting $(basename "$run_dir")"
        python scripts/export_paper_artifacts.py --run-dir "${run_dir%/}" || true
    fi
done

echo "--- exporting qualitative traces ---"
declare -A TRACE_CONFIGS=(
    ["fixture"]="data/fixtures/qa.jsonl:data/fixtures/frames.jsonl"
    ["nextgqa_timestamp_150"]="data/processed/nextgqa_val_150.jsonl:data/indices/nextgqa_timestamp_frames.jsonl"
)
for ds in "${!TRACE_CONFIGS[@]}"; do
    IFS=':' read -r qa frames <<< "${TRACE_CONFIGS[$ds]}"
    if [[ -f "$qa" && -f "$frames" ]]; then
        for run_dir in runs/zero_shot/${ds}__*/; do
            if [[ -d "$run_dir" ]]; then
                echo "  traces for $(basename "${run_dir%/}")"
                python scripts/export_qualitative_traces.py \
                    --run-dir "${run_dir%/}" \
                    --qa-path "$qa" \
                    --frames-path "$frames" \
                    --policy scripted_curator \
                    --max-examples 5 || true
            fi
        done
    fi
done

echo "--- run summary ---"
python -c "
import json
from pathlib import Path
for summary_path in sorted(Path('runs/zero_shot').glob('*/summary.json')):
    try:
        s = json.load(open(summary_path))
        groups = s.get('groups', {})
        if groups:
            print(f'\n{summary_path.parent.name}:')
            for key in sorted(groups):
                g = groups[key]
                print(f'  {key}: acc={g[\"accuracy\"]:.3f} frames={g[\"avg_frames\"]:.1f} fep={g.get(\"fep\",\"n/a\")}')
    except Exception as e:
        print(f'  ERROR reading {summary_path}: {e}')
"

echo "=== [$SLURM_JOB_ID] artifact export finished at $(date) ==="
