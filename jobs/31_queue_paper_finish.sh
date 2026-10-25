#!/usr/bin/env bash
# Queue all remaining inference for paper freeze: MLVU learned rerun + LVB/MLVU ablations.
# Submits in parallel across a40/l40; final paper refresh waits for all jobs.
set -euo pipefail
ROOT="/home/vis-comp/22b3929/vidcurator"
cd "$ROOT"
mkdir -p logs runs

PY="${HOME}/miniconda3/envs/vidcurator/bin/python"
echo "=== paper-finish queue $(date) ==="
squeue -u "$USER" 2>/dev/null || true
echo

JOB_IDS=()

queue_job() {
  local label="$1" partition="$2" qos="$3" run_dir="$4" job_script="$5"
  shift 5
  if [[ -n "$run_dir" && -f "$run_dir/summary.json" ]]; then
    echo "  SKIP $label (summary exists)"
    return 0
  fi
  if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qF "$label"; then
    echo "  SKIP $label (already queued)"
    return 0
  fi
  local jid
  jid=$(sbatch --partition="$partition" --qos="$qos" --job-name="$label" "$@" "$job_script" | awk '{print $NF}')
  echo "  QUEUED $label -> $jid ($partition)"
  JOB_IDS+=("$jid")
}

# Broken MLVU learned run used stem/index mismatch — wipe and rerun after code fix.
MLVU_LGRPO="runs/gpu_mlvu_qwen2_5_learned_grpo"
if [[ -f "$MLVU_LGRPO/summary.json" ]]; then
  avg_frames=$($PY -c "
import json
s=json.load(open('$MLVU_LGRPO/summary.json'))
g=s['groups'].get('learned_curator@8',{})
print(g.get('avg_frames', -1))
" 2>/dev/null || echo "-1")
  if [[ "$avg_frames" == "0.0" || "$avg_frames" == "0" ]]; then
    echo "[mlvu-learned] removing invalid run (avg_frames=0)"
    rm -rf "$MLVU_LGRPO/summary.json" "$MLVU_LGRPO/_shards" "$MLVU_LGRPO/_job_config_"*.json
  fi
fi

# --- MLVU learned GRPO rerun (a40, ~3h) ---
queue_job "vc-lgrpo-mlvu-rerun" a40 a40 "$MLVU_LGRPO" jobs/26_learned_grpo_eval.sh \
  --export=ALL,CONFIG_PATH=configs/gpu_mlvu_qwen2_5_learned_grpo.json,RUN_DIR="$MLVU_LGRPO",PORT=8045,QA_PATH=data/processed/mlvu_val_150.jsonl,FRAMES_PATH=data/indices/mlvu_150_siglip_frames.jsonl

# --- Zero-shot ablations: LVB + MLVU × Qwen2.5/Qwen3 (parallel) ---
queue_job "vc-zs-lvb-q25" a40 a40 runs/gpu_longvideobench_qwen2_5_zero_shot jobs/12_zero_shot_single.sh \
  --export=ALL,CONFIG_PATH=configs/gpu_longvideobench_qwen2_5_zero_shot.json,MODEL_PATH=models/qwen2_5_vl_7b,QA_PATH=data/processed/longvideobench_val_150.jsonl,FRAMES_PATH=data/indices/longvideobench_siglip_frames.jsonl,RUN_DIR=runs/gpu_longvideobench_qwen2_5_zero_shot,PORT=8050,JOB_LABEL=lvb-q25-zs

queue_job "vc-zs-lvb-q3" l40 l40 runs/gpu_longvideobench_qwen3_zero_shot jobs/12_zero_shot_single.sh \
  --export=ALL,CONFIG_PATH=configs/gpu_longvideobench_qwen3_zero_shot.json,MODEL_PATH=models/qwen3_vl_8b_instruct,QA_PATH=data/processed/longvideobench_val_150.jsonl,FRAMES_PATH=data/indices/longvideobench_siglip_frames.jsonl,RUN_DIR=runs/gpu_longvideobench_qwen3_zero_shot,PORT=8051,JOB_LABEL=lvb-q3-zs

queue_job "vc-zs-mlvu-q25" a40 a40 runs/gpu_mlvu_qwen2_5_zero_shot jobs/12_zero_shot_single.sh \
  --export=ALL,CONFIG_PATH=configs/gpu_mlvu_qwen2_5_zero_shot.json,MODEL_PATH=models/qwen2_5_vl_7b,QA_PATH=data/processed/mlvu_val_150.jsonl,FRAMES_PATH=data/indices/mlvu_150_siglip_frames.jsonl,RUN_DIR=runs/gpu_mlvu_qwen2_5_zero_shot,PORT=8052,JOB_LABEL=mlvu-q25-zs

queue_job "vc-zs-mlvu-q3" l40 l40 runs/gpu_mlvu_qwen3_zero_shot jobs/12_zero_shot_single.sh \
  --export=ALL,CONFIG_PATH=configs/gpu_mlvu_qwen3_zero_shot.json,MODEL_PATH=models/qwen3_vl_8b_instruct,QA_PATH=data/processed/mlvu_val_150.jsonl,FRAMES_PATH=data/indices/mlvu_150_siglip_frames.jsonl,RUN_DIR=runs/gpu_mlvu_qwen3_zero_shot,PORT=8053,JOB_LABEL=mlvu-q3-zs

# --- Final paper refresh after all new jobs ---
if ((${#JOB_IDS[@]} > 0)); then
  dep=$(IFS=:; echo "${JOB_IDS[*]}")
  refresh_jid=$(sbatch --partition=l40 --qos=l40 \
    --dependency="afterok:${dep}" \
    --job-name=vc-paper-freeze \
    jobs/20_paper_refresh.sh | awk '{print $NF}')
  echo
  echo "[paper-freeze] queued $refresh_jid after jobs: ${JOB_IDS[*]}"
else
  echo
  echo "[paper-freeze] nothing new queued — running refresh now"
  sbatch --partition=l40 --qos=l40 --job-name=vc-paper-freeze jobs/20_paper_refresh.sh
fi

echo
echo "=== queued $(date) ==="
squeue -u "$USER" -o "%.8i %.22j %.2t %.10M %R" 2>/dev/null || true
