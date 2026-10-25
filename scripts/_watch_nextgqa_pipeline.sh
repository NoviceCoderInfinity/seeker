#!/bin/bash
set -uo pipefail
cd /home/vis-comp/22b3929/vidcurator
log(){ echo "$(date '+%F %T') $*"; }
log "watcher start; waiting for [ok] nextgqa in robust_dl log"
# wait until robust downloader reports nextgqa complete
while ! grep -q "\[ok\] nextgqa: complete" logs/robust_dl_datasets.log 2>/dev/null; do
  if ! pgrep -f "robust_download" >/dev/null 2>&1; then
    log "WARN: robust_download exited before nextgqa [ok]; checking once more"
    grep -q "\[ok\] nextgqa: complete" logs/robust_dl_datasets.log 2>/dev/null && break
    log "ERROR: nextgqa not complete and downloader gone; aborting watcher"; exit 1
  fi
  sleep 30
done
log "nextgqa download complete; mp4s=$(find data/raw/nextgqa/NExTVideo -name '*.mp4' | wc -l)"
DEC=$(sbatch --parsable --partition=a40 --qos=a40 \
  --export=ALL,QA_PATH=data/processed/nextgqa_val.jsonl,VIDEO_ROOT=data/raw/nextgqa/NExTVideo,OUT_ROOT=data/frames/nextgqa,DATASET_NAME=nextgqa,FPS=1.0 \
  jobs/02b_decode_index_full.sh)
log "submitted nextgqa decode job=$DEC"
ZS=$(sbatch --parsable --partition=a40 --qos=a40 --dependency=afterok:$DEC \
  --job-name=vc-zs-nextgqa-full \
  --export=ALL,CONFIG_PATH=configs/gpu_nextgqa_qwen2_5_full.json,MODEL_PATH=models/qwen2_5_vl_7b,QA_PATH=data/processed/nextgqa_val.jsonl,FRAMES_PATH=data/indices/nextgqa_siglip_frames.jsonl,RUN_DIR=runs/gpu_nextgqa_qwen2_5_full,PORT=8020,JOB_LABEL=nextgqa_full \
  jobs/12_zero_shot_single.sh)
log "submitted nextgqa full-val Qwen2.5 job=$ZS (afterok:$DEC); NEXTGQA_PIPELINE_QUEUED decode=$DEC zs=$ZS"
