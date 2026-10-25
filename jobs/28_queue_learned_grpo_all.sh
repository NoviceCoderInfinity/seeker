#!/usr/bin/env bash
# Queue post-GRPO eval on all Qwen2.5 zero-shot benchmarks (respects QOS submit limit via chaining).
set -euo pipefail
ROOT="/home/vis-comp/22b3929/vidcurator"
cd "$ROOT"
mkdir -p logs configs

PY="${HOME}/miniconda3/envs/vidcurator/bin/python"
"$PY" scripts/gen_learned_grpo_configs.py

cat > configs/gpu_nextgqa_qwen2_5_learned_grpo_analysis.json <<'EOF'
{
  "name": "gpu_nextgqa_qwen2_5_learned_grpo_analysis",
  "frames_path": "data/indices/nextgqa_siglip_frames.jsonl",
  "qa_path": "data/processed/nextgqa_val_150_siglip.jsonl",
  "output_dir": "runs/gpu_nextgqa_qwen2_5_learned_grpo_analysis",
  "budgets": [2, 4, 8, 16, 32],
  "policies": ["scripted_curator", "learned_curator"],
  "action_policy": {
    "model_path": "models/qwen2_5_vl_7b",
    "adapter_path": "runs/grpo/qwen2_5_vl_7b_grpo_lora/lora_adapter",
    "max_new_tokens": 384
  },
  "answerer": {
    "kind": "openai_compat",
    "endpoint": "http://localhost:8000",
    "model": "models/qwen2_5_vl_7b",
    "send_images": true,
    "max_images": 16,
    "timeout_seconds": 600
  },
  "policy": {
    "max_hops": 4,
    "retrieve_per_hop": 6,
    "seed_uniform": 2,
    "sufficiency_threshold": 0.72
  },
  "retrieval_recall_at_k": [4, 8, 16, 32]
}
EOF

submit_one() {
  local label="$1" cfg="$2" run_dir="$3" port="$4" qa="$5" frames="$6" dep="${7:-}"
  if [[ -f "$run_dir/summary.json" ]]; then
    echo "  SKIP $label (done)"
    return 0
  fi
  local args=(--partition=a40 --qos=a40 --job-name="vc-lgrpo-${label}")
  [[ -n "$dep" ]] && args+=(--dependency="afterok:$dep")
  args+=(--export=ALL,CONFIG_PATH="$cfg",RUN_DIR="$run_dir",PORT="$port",QA_PATH="$qa",FRAMES_PATH="$frames")
  local jid
  jid=$(sbatch "${args[@]}" jobs/26_learned_grpo_eval.sh | awk '{print $NF}')
  echo "  QUEUED $label -> $jid (dep=${dep:-none})"
  echo "$jid"
}

echo "=== learned GRPO queue $(date) ==="
squeue -u "$USER" 2>/dev/null || true
echo

LAST=""
chain() {
  local label="$1" cfg="$2" run_dir="$3" port="$4" qa="$5" frames="$6"
  if [[ -f "$run_dir/summary.json" ]]; then
    echo "[$label] done"
    return 0
  fi
  echo "[$label]"
  local jid
  jid=$(submit_one "$label" "$cfg" "$run_dir" "$port" "$qa" "$frames" "$LAST")
  [[ -n "$jid" ]] && LAST="$jid"
}

# If NExT-GQA still running, chain starts after it for patch/resume
if squeue -u "$USER" -h -j 144486 &>/dev/null; then
  LAST="144486"
  echo "[nextgqa] active job 144486 — chain continues after it"
fi

# Skip standalone nextgqa if 144486 running; patch job fills missing budgets via shard resume
if [[ "$LAST" == "144486" ]]; then
  echo "[nextgqa-patch] will resume b2/b32 after 144486"
  jid=$(submit_one ngqa-patch configs/gpu_nextgqa_qwen2_5_learned_grpo_analysis.json \
    runs/gpu_nextgqa_qwen2_5_learned_grpo_analysis 8046 \
    data/processed/nextgqa_val_150_siglip.jsonl data/indices/nextgqa_siglip_frames.jsonl "$LAST")
  [[ -n "$jid" ]] && LAST="$jid"
fi

# Only queue datasets not already submitted
declare -A SPECS=(
  [nextqa]="configs/gpu_nextqa_qwen2_5_learned_grpo.json|runs/gpu_nextqa_qwen2_5_learned_grpo|8041|data/processed/nextqa_temporal_val_150.jsonl|data/indices/nextqa_temporal_siglip_frames.jsonl"
  [egoschema]="configs/gpu_egoschema_qwen2_5_learned_grpo.json|runs/gpu_egoschema_qwen2_5_learned_grpo|8042|data/processed/egoschema_val_150.jsonl|data/indices/egoschema_siglip_frames.jsonl"
  [videomme]="configs/gpu_videomme_qwen2_5_learned_grpo.json|runs/gpu_videomme_qwen2_5_learned_grpo|8043|data/processed/videomme_long_150.jsonl|data/indices/videomme_long_siglip_frames.jsonl"
  [longvideobench]="configs/gpu_longvideobench_qwen2_5_learned_grpo.json|runs/gpu_longvideobench_qwen2_5_learned_grpo|8044|data/processed/longvideobench_val_150.jsonl|data/indices/longvideobench_150_siglip_frames.jsonl"
  [mlvu]="configs/gpu_mlvu_qwen2_5_learned_grpo.json|runs/gpu_mlvu_qwen2_5_learned_grpo|8045|data/processed/mlvu_val_150.jsonl|data/indices/mlvu_150_siglip_frames.jsonl"
)

for ds in nextqa egoschema videomme longvideobench mlvu; do
  IFS='|' read -r cfg run_dir port qa frames <<< "${SPECS[$ds]}"
  # skip if already in queue (job 144504-507 may exist)
  if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -q "vc-lgrpo-${ds}"; then
    echo "[$ds] already queued"
    continue
  fi
  chain "$ds" "$cfg" "$run_dir" "$port" "$qa" "$frames"
done

if [[ -n "$LAST" ]]; then
  scancel 144488 2>/dev/null || true
  echo
  echo "[final] paper refresh after job $LAST"
  sbatch --partition=l40 --qos=l40 --dependency="afterok:$LAST" --job-name=vc-paper-final jobs/20_paper_refresh.sh
fi

echo
squeue -u "$USER" 2>/dev/null || true
