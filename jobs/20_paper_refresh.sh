#!/bin/bash
#SBATCH --job-name=vc-paper-refresh
#SBATCH --output=logs/paper_refresh_%j.out
#SBATCH --error=logs/paper_refresh_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Regenerate paper tables + figures from completed runs/ summaries.
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs reports figures

PY="${HOME}/miniconda3/envs/vidcurator/bin/python"
echo "=== paper refresh $(date) ==="
echo "Completed runs: $(ls runs/gpu_*/summary.json 2>/dev/null | wc -l)"

# Merge parallel shard workers into full summaries before aggregation.
for cfg in \
  configs/gpu_longvideobench_qwen2_5_zero_shot.json \
  configs/gpu_mlvu_qwen2_5_zero_shot.json \
  configs/gpu_mlvu_qwen3_zero_shot.json; do
  if [[ -f "$cfg" ]]; then
    "$PY" scripts/finalize_run_from_shards.py --config "$cfg" --force || true
  fi
done

"$PY" scripts/aggregate_results.py
"$PY" scripts/make_figures.py || echo "WARN: make_figures partial"

echo "=== outputs ==="
ls -la reports/*.csv reports/summary.md reports/pareto_all.md 2>/dev/null | head -20
echo "=== done $(date) ==="
