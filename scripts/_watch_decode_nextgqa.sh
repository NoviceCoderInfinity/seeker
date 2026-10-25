#!/bin/bash
set -uo pipefail
cd /home/vis-comp/22b3929/vidcurator
# wait for nextgqa download to finish
while pgrep -f "download_priority_assets.*nextgqa" >/dev/null 2>&1; do sleep 30; done
echo "$(date) nextgqa download finished; mp4s=$(find data/raw/nextgqa/NExTVideo -name '*.mp4' | wc -l)"
sbatch --partition=a40 --qos=a40 \
  --export=ALL,QA_PATH=data/processed/nextgqa_val.jsonl,VIDEO_ROOT=data/raw/nextgqa/NExTVideo,OUT_ROOT=data/frames/nextgqa,DATASET_NAME=nextgqa,FPS=1.0 \
  jobs/02b_decode_index_full.sh
echo "$(date) submitted nextgqa full decode job"
