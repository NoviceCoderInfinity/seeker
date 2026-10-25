#!/usr/bin/env bash
# Queue pending inference respecting QOS: a40 max 2 GPU, l40 max 4 GPU, dgx for Qwen3 @16.
# Keeps healthy in-flight jobs; cancels failed/blocked vc-* jobs only.
set -euo pipefail
ROOT="/home/vis-comp/22b3929/vidcurator"
cd "$ROOT"
mkdir -p logs runs

echo "=== queue-all-pending $(date) ==="

# Cancel failed/blocked paper-finish jobs; keep healthy runners (e.g. slow LVB Q2.5).
for j in $(squeue -u "$USER" -h -o "%i %j %T" 2>/dev/null | awk '
  /vc-paper-freeze/ {print $1}
  /vc-lgrpo|vc-zs-/ && /PENDING/ && $0 ~ /DependencyNeverSatisfied/ {print $1}
'); do
  echo "  scancel blocked $j"
  scancel "$j" 2>/dev/null || true
done
# Also cancel any failed vc-* still showing in queue
for j in $(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk '/vc-lgrpo|vc-zs-|vc-paper-freeze/ {print $1}'); do
  state=$(sacct -j "$j" --format=State -P -n 2>/dev/null | head -1 | cut -d'|' -f1)
  if [[ "$state" == "FAILED" ]]; then
    echo "  scancel failed $j"
    scancel "$j" 2>/dev/null || true
  fi
done
sleep 2

JOB_IDS=()

# Track already-running healthy jobs for paper-freeze dependency.
for j in $(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk '/vc-zs-lvb-q25|vc-zs-mlvu-q25/ {print $1}'); do
  JOB_IDS+=("$j")
  echo "  KEEP running job $j"
done

submit() {
  local label="$1" partition="$2" qos="$3" run_dir="$4" script="$5"
  shift 5
  if [[ -f "$run_dir/summary.json" ]]; then
    echo "  SKIP $label (done)"
    return 0
  fi
  local n=0
  [[ -d "$run_dir/_shards" ]] && n=$(ls "$run_dir/_shards"/*.jsonl 2>/dev/null | wc -l)
  local jid
  jid=$(sbatch --partition="$partition" --qos="$qos" --gres=gpu:1 \
    --job-name="$label" "$@" "$script" | awk '{print $NF}')
  echo "  QUEUED $label -> $jid ($partition, ${n} shards cached)"
  JOB_IDS+=("$jid")
}

MLVU_LGRPO="runs/gpu_mlvu_qwen2_5_learned_grpo"
rm -f "$MLVU_LGRPO/summary.json"

# MLVU learned GRPO — action policy on CPU avoids dual-GPU OOM on A40.
if [[ ! -f "$MLVU_LGRPO/summary.json" ]]; then
  submit "vc-lgrpo-mlvu" a40 a40 "$MLVU_LGRPO" jobs/26_learned_grpo_eval.sh \
    --export=ALL,ACTION_POLICY_DEVICE=cpu,CONFIG_PATH=configs/gpu_mlvu_qwen2_5_learned_grpo.json,RUN_DIR="$MLVU_LGRPO",PORT=8045,QA_PATH=data/processed/mlvu_val_150.jsonl,FRAMES_PATH=data/indices/mlvu_150_siglip_frames.jsonl
fi
LGRPO_JID="${JOB_IDS[-1]:-}"

# Qwen3 zero-shot on dgx (80GB) — L40 OOMs at budget 16 with 16 images.
if [[ ! -f runs/gpu_mlvu_qwen3_zero_shot/summary.json ]]; then
  submit "vc-zs-mlvu-q3" dgx dgx runs/gpu_mlvu_qwen3_zero_shot jobs/12_zero_shot_single.sh \
    --export=ALL,CONFIG_PATH=configs/gpu_mlvu_qwen3_zero_shot.json,MODEL_PATH=models/qwen3_vl_8b_instruct,QA_PATH=data/processed/mlvu_val_150.jsonl,FRAMES_PATH=data/indices/mlvu_150_siglip_frames.jsonl,RUN_DIR=runs/gpu_mlvu_qwen3_zero_shot,PORT=8053,JOB_LABEL=mlvu-q3-zs
fi

# LVB Q2.5 — skip if already running (144851 etc.)
if [[ ! -f runs/gpu_longvideobench_qwen2_5_zero_shot/summary.json ]]; then
  running_lvb=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c 'vc-zs-lvb-q25' || true)
  if [[ "$running_lvb" -eq 0 ]]; then
    submit "vc-zs-lvb-q25" a40 a40 runs/gpu_longvideobench_qwen2_5_zero_shot jobs/12_zero_shot_single.sh \
      --export=ALL,CONFIG_PATH=configs/gpu_longvideobench_qwen2_5_zero_shot.json,MODEL_PATH=models/qwen2_5_vl_7b,QA_PATH=data/processed/longvideobench_val_150.jsonl,FRAMES_PATH=data/indices/longvideobench_siglip_frames.jsonl,RUN_DIR=runs/gpu_longvideobench_qwen2_5_zero_shot,PORT=8050,JOB_LABEL=lvb-q25-zs
  else
    echo "  SKIP vc-zs-lvb-q25 (already running)"
  fi
fi

# MLVU Q2.5 — chain after learned GRPO frees an a40 slot (if learned was queued).
if [[ ! -f runs/gpu_mlvu_qwen2_5_zero_shot/summary.json ]]; then
  running_mlvu25=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c 'vc-zs-mlvu-q25' || true)
  if [[ "$running_mlvu25" -eq 0 ]]; then
    n=$(ls runs/gpu_mlvu_qwen2_5_zero_shot/_shards/*.jsonl 2>/dev/null | wc -l)
    dep_arg=()
    [[ -n "${LGRPO_JID:-}" ]] && dep_arg=(--dependency="afterok:${LGRPO_JID}")
    jid=$(sbatch --partition=a40 --qos=a40 --gres=gpu:1 \
      "${dep_arg[@]}" \
      --job-name=vc-zs-mlvu-q25 \
      --export=ALL,CONFIG_PATH=configs/gpu_mlvu_qwen2_5_zero_shot.json,MODEL_PATH=models/qwen2_5_vl_7b,QA_PATH=data/processed/mlvu_val_150.jsonl,FRAMES_PATH=data/indices/mlvu_150_siglip_frames.jsonl,RUN_DIR=runs/gpu_mlvu_qwen2_5_zero_shot,PORT=8052,JOB_LABEL=mlvu-q25-zs \
      jobs/12_zero_shot_single.sh | awk '{print $NF}')
    echo "  QUEUED vc-zs-mlvu-q25 -> $jid (a40${LGRPO_JID:+, after $LGRPO_JID}, ${n} shards)"
    JOB_IDS+=("$jid")
  else
    # Replace stale dependency chains from prior failed lgrpo jobs.
    stale=$(squeue -u "$USER" -h -o "%i %j %R" 2>/dev/null | awk '/vc-zs-mlvu-q25/ && /DependencyNeverSatisfied/ {print $1}')
    if [[ -n "$stale" && -n "${LGRPO_JID:-}" ]]; then
      echo "  scancel stale vc-zs-mlvu-q25 $stale"
      scancel "$stale" 2>/dev/null || true
      n=$(ls runs/gpu_mlvu_qwen2_5_zero_shot/_shards/*.jsonl 2>/dev/null | wc -l)
      jid=$(sbatch --partition=a40 --qos=a40 --gres=gpu:1 \
        --dependency="afterok:${LGRPO_JID}" \
        --job-name=vc-zs-mlvu-q25 \
        --export=ALL,CONFIG_PATH=configs/gpu_mlvu_qwen2_5_zero_shot.json,MODEL_PATH=models/qwen2_5_vl_7b,QA_PATH=data/processed/mlvu_val_150.jsonl,FRAMES_PATH=data/indices/mlvu_150_siglip_frames.jsonl,RUN_DIR=runs/gpu_mlvu_qwen2_5_zero_shot,PORT=8052,JOB_LABEL=mlvu-q25-zs \
        jobs/12_zero_shot_single.sh | awk '{print $NF}')
      echo "  REQUEUED vc-zs-mlvu-q25 -> $jid (after $LGRPO_JID, ${n} shards)"
      JOB_IDS+=("$jid")
    else
      echo "  SKIP vc-zs-mlvu-q25 (already queued/running)"
    fi
  fi
fi

if ((${#JOB_IDS[@]} > 0)); then
  dep=$(IFS=:; echo "${JOB_IDS[*]}")
  ref=$(sbatch --partition=l40 --qos=l40 --dependency="afterok:${dep}" \
    --job-name=vc-paper-freeze jobs/20_paper_refresh.sh | awk '{print $NF}')
  echo "[paper-freeze] $ref after ${JOB_IDS[*]}"
else
  echo "All runs complete — refreshing reports"
  sbatch --partition=l40 --qos=l40 --job-name=vc-paper-freeze jobs/20_paper_refresh.sh
fi

echo
squeue -u "$USER" -o "%.8i %.22j %.2t %.10M %R" 2>/dev/null || true
