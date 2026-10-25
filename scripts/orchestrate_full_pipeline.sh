#!/bin/bash
# Hands-off: extract videos -> decode+SigLIP-index full split -> full-val Qwen2.5,
# for every remaining dataset. Respects a40 QoS (<=6 submitted). Resumable + idempotent.
set -uo pipefail
cd /home/vis-comp/22b3929/vidcurator
export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
PY="$HOME/miniconda3/envs/vidcurator/bin/python"
log(){ echo "$(date '+%F %T') $*"; }

A40_CAP=6   # MaxSubmitJobsPerUser for a40

wait_for_slot(){ # ensure we can submit $1 more a40 jobs without exceeding cap
  local need="$1"
  while :; do
    local n; n=$(squeue -u "$USER" -h -p a40 2>/dev/null | wc -l)
    if (( n + need <= A40_CAP )); then return 0; fi
    log "a40 slots full ($n submitted); waiting 120s..."
    sleep 120
  done
}

extract_videomme(){
  [[ $(find data/videos/videomme -iname '*.mp4' 2>/dev/null | wc -l) -ge 250 ]] && { log "videomme already extracted"; return 0; }
  log "extracting videomme zips..."
  $PY scripts/extract_video_archives.py --dataset videomme >> logs/extract_videomme.log 2>&1
  log "videomme mp4s=$(find data/videos/videomme -iname '*.mp4' | wc -l)"
}
extract_lvb(){
  mkdir -p data/videos/longvideobench
  [[ $(find data/videos/longvideobench -iname '*.mp4' 2>/dev/null | wc -l) -ge 1000 ]] && { log "lvb already extracted"; return 0; }
  log "extracting longvideobench tar parts..."
  cat data/raw/longvideobench/videos.tar.part.* | tar -xf - -C data/videos/longvideobench >> logs/extract_lvb.log 2>&1
  log "lvb mp4s=$(find data/videos/longvideobench -iname '*.mp4' | wc -l)"
}
extract_mlvu(){
  mkdir -p data/videos/mlvu
  [[ $(find data/videos/mlvu -iname '*.mp4' 2>/dev/null | wc -l) -ge 100 ]] && { log "mlvu already extracted"; return 0; }
  log "extracting mlvu tar.gz parts..."
  cat data/raw/mlvu/MLVU_Test/test_video.tar.gz.part-* | tar -xzf - -C data/videos/mlvu >> logs/extract_mlvu.log 2>&1
  log "mlvu mp4s=$(find data/videos/mlvu -iname '*.mp4' | wc -l)"
}

# dataset | qa | video_root | out_root | index_name | fps | config | port
process(){
  local ds="$1" qa="$2" vroot="$3" out="$4" name="$5" fps="$6" cfg="$7" port="$8"
  if [[ -f "runs/gpu_${ds}_qwen2_5_full/summary.json" ]]; then log "$ds full-val already done; skip"; return 0; fi
  # force full re-decode: drop stale pilot-sized decoded index
  rm -f "data/indices/${name}_decoded_frames.jsonl"
  wait_for_slot 2
  local DEC; DEC=$(sbatch --parsable --partition=a40 --qos=a40 \
    --export=ALL,QA_PATH="$qa",VIDEO_ROOT="$vroot",OUT_ROOT="$out",DATASET_NAME="$name",FPS="$fps" \
    jobs/02b_decode_index_full.sh)
  local ZS; ZS=$(sbatch --parsable --partition=a40 --qos=a40 --dependency=afterok:$DEC \
    --job-name="vc-zs-${ds}-full" \
    --export=ALL,CONFIG_PATH="configs/${cfg}",MODEL_PATH=models/qwen2_5_vl_7b,QA_PATH="$qa",FRAMES_PATH="data/indices/${name}_siglip_frames.jsonl",RUN_DIR="runs/gpu_${ds}_qwen2_5_full",PORT="$port",JOB_LABEL="${ds}_full" \
    jobs/12_zero_shot_single.sh)
  log "$ds: decode=$DEC fullval=$ZS (afterok)"
}

log "=== orchestrator start ==="
# 1) datasets whose videos are already on disk
process nextqa     data/processed/nextqa_temporal_test.jsonl data/raw/nextgqa/NExTVideo data/frames/nextqa     nextqa_temporal 1.0 gpu_nextqa_qwen2_5_full.json     8030
process egoschema  data/processed/egoschema_val.jsonl        data/videos/egoschema/videos data/frames/egoschema egoschema       0.5 gpu_egoschema_qwen2_5_full.json  8031
# 2) extract + process archived datasets
extract_videomme; process videomme data/processed/videomme_long.jsonl data/videos/videomme data/frames/videomme videomme_long 0.1 gpu_videomme_qwen2_5_full.json 8032
extract_mlvu;      process mlvu     data/processed/mlvu_val.jsonl       data/videos/mlvu     data/frames/mlvu     mlvu          0.1 gpu_mlvu_qwen2_5_full.json     8034
extract_lvb;       process longvideobench data/processed/longvideobench_val.jsonl data/videos/longvideobench data/frames/longvideobench longvideobench 0.1 gpu_longvideobench_qwen2_5_full.json 8033
log "=== orchestrator: all datasets submitted ==="
