#!/usr/bin/env bash
# Submit Qwen3 full jobs that hit QOSMaxSubmitJobPerUserLimit during initial queue burst.
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator
mkdir -p logs

submit() {
  local cfg="$1" qa="$2" frames="$3" run_dir="$4" port="$5"
  if [[ -f "$run_dir/summary.json" ]]; then
    echo "SKIP (done): $cfg"
    return 0
  fi
  sbatch --partition=l40 --qos=l40 \
    --export=ALL,CONFIG_PATH=configs/${cfg}.json,MODEL_PATH=models/qwen3_vl_8b_instruct,QA_PATH=$qa,FRAMES_PATH=$frames,RUN_DIR=$run_dir,PORT=$port,JOB_LABEL=$cfg \
    jobs/12_zero_shot_single.sh
}

submit gpu_nextgqa_qwen3_full \
  data/processed/nextgqa_val_siglip.jsonl \
  data/indices/nextgqa_siglip_frames.jsonl \
  runs/gpu_nextgqa_qwen3_full 8024

submit gpu_nextqa_qwen3_full \
  data/processed/nextqa_temporal_test_3500_siglip.jsonl \
  data/indices/nextqa_temporal_siglip_frames.jsonl \
  runs/gpu_nextqa_qwen3_full 8025

squeue -u "$USER"
