#!/bin/bash
set -uo pipefail
cd /home/vis-comp/22b3929/vidcurator
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your environment; never hard-code tokens}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
PY="$HOME/miniconda3/envs/vidcurator/bin/python"
# wait for nextgqa download to finish to avoid bandwidth contention
while pgrep -f "download_priority_assets.*nextgqa" >/dev/null 2>&1; do sleep 30; done
for ds in videomme_long longvideobench mlvu; do
  echo "=== $(date) downloading $ds ==="
  $PY scripts/download_priority_assets.py --datasets --only "$ds" --allow-large-video
  echo "=== $(date) done $ds (exit $?) ==="
done
echo "ALL_REMAINING_DOWNLOADS_DONE"
