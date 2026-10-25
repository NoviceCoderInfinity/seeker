#!/bin/bash
# Submit all 6 R1 v2 re-baseline pilots (fixed harness) across dgx + l40.
# Run ONLY after Gate G0 passes (sanity confirms uniform is monotone).
# Each job: unique PORT, own RUN_DIR, Qwen2.5. dgx (A100 80G) preferred; l40 fallback.
set -euo pipefail
cd /home/vis-comp/22b3929/vidcurator

# dataset : qa_150 : frames_index : partition : port
JOBS=(
  "nextgqa|data/processed/nextgqa_val_150.jsonl|data/indices/nextgqa_siglip_frames.jsonl|dgx|8060"
  "nextqa|data/processed/nextqa_temporal_val_150.jsonl|data/indices/nextqa_temporal_siglip_frames.jsonl|dgx|8061"
  "videomme|data/processed/videomme_long_150.jsonl|data/indices/videomme_long_siglip_frames.jsonl|dgx|8062"
  "egoschema|data/processed/egoschema_val_150.jsonl|data/indices/egoschema_siglip_frames.jsonl|dgx|8063"
  "longvideobench|data/processed/longvideobench_val_150.jsonl|data/indices/longvideobench_siglip_frames.jsonl|l40|8064"
  "mlvu|data/processed/mlvu_val_150.jsonl|data/indices/mlvu_150_siglip_frames.jsonl|l40|8065"
)

for entry in "${JOBS[@]}"; do
  IFS='|' read -r name qa frames part port <<< "$entry"
  cfg="configs/r1_v2_${name}_qwen2_5.json"
  run="runs/r1_v2_${name}_qwen2_5"
  echo "submitting R1 $name on $part port $port"
  sbatch --partition="$part" --qos="$part" --gres=gpu:1 \
    --job-name="vc-r1-${name}" \
    --export=ALL,CONFIG_PATH="$cfg",MODEL_PATH=models/qwen2_5_vl_7b,QA_PATH="$qa",FRAMES_PATH="$frames",RUN_DIR="$run",PORT="$port",JOB_LABEL="r1-${name}" \
    jobs/12_zero_shot_single.sh
done
echo "all R1 v2 pilots submitted."
