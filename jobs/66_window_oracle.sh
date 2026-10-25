#!/bin/bash
#SBATCH --output=logs/winor_%j.out
#SBATCH --error=logs/winor_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# G2 gate: temporal-window oracle on hour-long sets (plan8 memory program).
set -uo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs reports
export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
export HF_HOME="/home/vis-comp/22b3929/vidcurator/.hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLM_MAX_PIXELS="${VLM_MAX_PIXELS:-200704}"
export PYTHONUNBUFFERED=1

PORT=8563
python scripts/serve_transformers_vlm.py --model models/qwen2_5_vl_7b --port "$PORT" --dtype bfloat16 \
  > "logs/vlm_${SLURM_JOB_ID}_winor.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null' EXIT
for attempt in $(seq 1 180); do
  python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5)" 2>/dev/null && break
  sleep 5
done

FAIL=0
[[ -f reports/window_oracle_lvbench.md ]] || python scripts/window_oracle.py \
  --qa data/processed/heldout/lvbench_heldout.jsonl \
  --frames data/indices/lvbench_siglip_frames.jsonl \
  --endpoint "http://127.0.0.1:${PORT}" --windows 8 --per-window 8 \
  --out reports/window_oracle_lvbench.md || FAIL=1
[[ -f reports/window_oracle_videomme.md ]] || python scripts/window_oracle.py \
  --qa data/processed/videomme_long_150.jsonl \
  --frames data/indices/videomme_long_siglip_frames.jsonl \
  --endpoint "http://127.0.0.1:${PORT}" --windows 8 --per-window 8 \
  --out reports/window_oracle_videomme.md || FAIL=1
echo "=== [$SLURM_JOB_ID] window-oracle done (fail=$FAIL) ==="
exit "$FAIL"
