#!/usr/bin/env bash
# Launch parallel zero-shot / shard workers on dgx (QOS allows 4 GPU; a40 capped at 2).
set -euo pipefail
ROOT="/home/vis-comp/22b3929/vidcurator"
cd "$ROOT"

echo "=== parallel-finish $(date) ==="
squeue -u "$USER" -o "%.8i %.22j %.2t %.10M %R" || true

# Drop blocked pending jobs; keep healthy runners.
scancel 144931 144932 2>/dev/null || true
sleep 2

JOB_IDS=()
for j in $(squeue -u "$USER" -h -o "%i %j %T" 2>/dev/null | awk '/vc-zs-|vc-lgrpo|vc-paper/ {print $1}'); do
  JOB_IDS+=("$j")
done

submit_dgx() {
  local label="$1" config="$2" port="$3" shards_only="${4:-0}"
  local extra=""
  [[ "$shards_only" == "1" ]] && extra="EVAL_SHARDS_ONLY=1,"
  local jid
  jid=$(sbatch --partition=dgx --qos=dgx --gres=gpu:1 \
    --job-name="$label" \
    --export=ALL,${extra}CONFIG_PATH="$config",MODEL_PATH=models/qwen2_5_vl_7b,QA_PATH=data/processed/longvideobench_val_150.jsonl,FRAMES_PATH=data/indices/longvideobench_siglip_frames.jsonl,RUN_DIR=runs/gpu_longvideobench_qwen2_5_zero_shot,PORT="$port",JOB_LABEL="$label" \
    jobs/12_zero_shot_single.sh 2>&1 | awk '{print $NF}')
  echo "  QUEUED $label -> $jid (dgx port $port shards_only=$shards_only)"
  JOB_IDS+=("$jid")
}

# MLVU Q2.5 — was wrongly chained to a40; run on dgx now.
if [[ ! -f runs/gpu_mlvu_qwen2_5_zero_shot/summary.json ]]; then
  jid=$(sbatch --partition=dgx --qos=dgx --gres=gpu:1 \
    --job-name=vc-zs-mlvu-q25 \
    --export=ALL,CONFIG_PATH=configs/gpu_mlvu_qwen2_5_zero_shot.json,MODEL_PATH=models/qwen2_5_vl_7b,QA_PATH=data/processed/mlvu_val_150.jsonl,FRAMES_PATH=data/indices/mlvu_150_siglip_frames.jsonl,RUN_DIR=runs/gpu_mlvu_qwen2_5_zero_shot,PORT=8054,JOB_LABEL=mlvu-q25-zs \
    jobs/12_zero_shot_single.sh | awk '{print $NF}')
  echo "  QUEUED vc-zs-mlvu-q25 -> $jid (dgx)"
  JOB_IDS+=("$jid")
fi

# LVB Q2.5 — split across dgx workers (drop monolithic a40 job if still running).
if [[ ! -f runs/gpu_longvideobench_qwen2_5_zero_shot/summary.json ]]; then
  submit_dgx "vc-zs-lvb-cur" configs/gpu_longvideobench_qwen2_5_zero_shot_curators.json 8055 1
  submit_dgx "vc-zs-lvb-ret" configs/gpu_longvideobench_qwen2_5_zero_shot_retrieval.json 8056 1
  scancel $(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk '$2=="vc-zs-lvb-q25"{print $1}') 2>/dev/null || true
fi

# Paper refresh after all eval jobs (+ finalize step inside refresh script).
if ((${#JOB_IDS[@]} > 0)); then
  dep=$(IFS=:; echo "${JOB_IDS[*]}")
  # Also wait on existing runners if still up.
  for j in 144851 144928 144929; do
    if squeue -j "$j" -h 2>/dev/null | grep -q .; then
      dep="${dep}:${j}"
    fi
  done
  ref=$(sbatch --partition=l40 --qos=l40 --dependency="afterok:${dep}" \
    --job-name=vc-paper-freeze jobs/20_paper_refresh.sh | awk '{print $NF}')
  echo "[paper-freeze] $ref after ${dep}"
fi

echo
squeue -u "$USER" -o "%.8i %.22j %.2t %.10M %R" || true
