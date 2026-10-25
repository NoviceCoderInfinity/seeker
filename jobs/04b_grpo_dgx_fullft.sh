#!/bin/bash
#SBATCH --job-name=vidcurator-grpo-fullft
#SBATCH --output=logs/grpo_fullft_%j.out
#SBATCH --error=logs/grpo_fullft_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32

# Full fine-tune GRPO variant of jobs/04_grpo_dgx.sh.
# Differences vs the LoRA job:
#   - 8 GPUs (full 7B optimizer states need the extra shards + CPU offload).
#   - lora_rank=0 (verl treats 0 as full fine-tuning).
#   - FSDP2 param_offload + optimizer_offload + gradient checkpointing (memory).
#   - No SFT cold-start adapter (that is a LoRA concept); trains from base weights.
# Runs in PARALLEL with the LoRA run on a separate dgx node; keep whichever eval wins.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs/grpo

if [[ -d "$HOME/miniconda3/envs/vidcurator-grpo" ]]; then
    export PATH="$HOME/miniconda3/envs/vidcurator-grpo/bin:$PATH"
elif [[ -d "$HOME/miniconda3/envs/vidcurator" ]]; then
    export PATH="$HOME/miniconda3/envs/vidcurator/bin:$PATH"
elif [[ -f .conda-env/bin/activate ]]; then
    source .conda-env/bin/activate
elif [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
else
    echo "ERROR: no Python environment found" >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONSTARTUP="${SLURM_SUBMIT_DIR}/scripts/verl_vllm_compat.py"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset PYTORCH_CUDA_ALLOC_CONF
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0

export RAY_TMPDIR="/tmp/ray_${USER}_${SLURM_JOB_ID:-local}"
mkdir -p "$RAY_TMPDIR"
export RAY_DISABLE_IMPORT_WARNING=1
ray stop --force 2>/dev/null || true

echo "GPU env after cleanup: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"

MODEL_INDEX="${MODEL_INDEX:-0}"
GRPO_CONFIG="${GRPO_CONFIG:-configs/grpo_stage1_fullft.json}"

echo "--- GRPO environment preflight ---"
python scripts/verify_grpo_env.py \
    --repo "$SLURM_SUBMIT_DIR" \
    --config "$GRPO_CONFIG" \
    --model-index "$MODEL_INDEX" \
    --skip-hydra

resolve_verl_root() {
    local candidate
    for candidate in \
        "$SLURM_SUBMIT_DIR/third_party/verl" \
        "${VERL_ROOT:-}"; do
        if [[ -n "$candidate" && -f "$candidate/examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

VERL_ROOT="$(resolve_verl_root || true)"
if [[ -z "${VERL_ROOT:-}" ]]; then
    echo "ERROR: verl entrypoint not found under third_party/verl or VERL_ROOT" >&2
    exit 1
fi

echo "=== [$SLURM_JOB_ID] FULL-FT GRPO training started at $(date) ==="
echo "Model index: $MODEL_INDEX"
echo "Config: $GRPO_CONFIG (FULL fine-tune, lora_rank=0)"
echo "VERL_ROOT: $VERL_ROOT"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"

TRAIN_DATA="${TRAIN_DATA:-data/rl/vidcurator_train_sft.jsonl}"
EVAL_DATA="${EVAL_DATA:-data/rl/vidcurator_val_sft.jsonl}"
if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "ERROR: training data not found: $TRAIN_DATA" >&2
    exit 1
fi

MODEL_PATH=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['base_models'][int('$MODEL_INDEX')])")
GLOBAL_BS=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['verl']['global_batch_size'])")
ROLLOUT_N=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['verl']['rollout_n'])")
TRAIN_MAX=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['verl'].get('train_max_samples', -1))")
VAL_MAX=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['verl'].get('val_max_samples', -1))")
ROLLOUT_TP=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['verl'].get('rollout_tp', 2))")
SAVE_FREQ=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['verl'].get('save_freq', 64))")
TEST_FREQ=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['verl'].get('test_freq', 64))")
OUTPUT_DIR="runs/grpo/$(basename "$MODEL_PATH")_fullft"

echo "--- preparing verl parquet ---"
python scripts/prep_grpo_parquet.py \
    --train "$TRAIN_DATA" \
    --val "$EVAL_DATA" \
    --out-dir data/rl/parquet

TRAIN_PARQUET="$SLURM_SUBMIT_DIR/data/rl/parquet/train.parquet"
EVAL_PARQUET="$SLURM_SUBMIT_DIR/data/rl/parquet/val.parquet"

VERL_ENTRYPOINT="$VERL_ROOT/examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh"

NGPUS="${SLURM_GPUS_ON_NODE:-8}"
RAY_CPUS="${SLURM_CPUS_PER_TASK:-32}"
export MODEL_PATH
export NGPUS_PER_NODE="$NGPUS"
export TRAIN_BATCH_SIZE="$GLOBAL_BS"
export PPO_MINI_BATCH_SIZE=$(( GLOBAL_BS / 2 ))
export ROLLOUT_N="$ROLLOUT_N"
export ROLLOUT_TP="$ROLLOUT_TP"
export TOTAL_EPOCHS="${GRPO_EPOCHS:-$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['verl'].get('epochs', 1))")}"
export SAVE_FREQ="${SAVE_FREQ}"
export TEST_FREQ="${TEST_FREQ}"
export PROJECT_NAME="${GRPO_PROJECT:-vidcurator_grpo}"
export EXPERIMENT_NAME="${GRPO_EXPERIMENT:-fullft_$(basename "$MODEL_PATH")}"
export MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT:-2048}"
export MAX_RESPONSE_LENGTH="${GRPO_MAX_RESPONSE:-256}"

# Full fine-tune: lora_rank=0 tells verl to train all weights. Memory is made to fit
# via FSDP2 full-shard + CPU offload of params & optimizer + gradient checkpointing.
GRPO_ARGS=(
    data.train_files="$TRAIN_PARQUET"
    data.val_files="$EVAL_PARQUET"
    data.prompt_key=prompt
    "data.train_max_samples=$TRAIN_MAX"
    "data.val_max_samples=$VAL_MAX"
    "data.val_batch_size=32"
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.lora_rank=0
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.actor.optim.lr=1e-6
    trainer.default_local_dir="$OUTPUT_DIR"
    data.train_batch_size="$GLOBAL_BS"
    actor_rollout_ref.rollout.n="$ROLLOUT_N"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096
    data.max_response_length=256
    trainer.n_gpus_per_node="$NGPUS"
    "trainer.total_epochs=$TOTAL_EPOCHS"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=$TEST_FREQ"
    trainer.val_before_train=false
    "reward.custom_reward_function.path=$SLURM_SUBMIT_DIR/scripts/vidcurator_verl_reward.py"
    reward.custom_reward_function.name=compute_score
    'trainer.logger=["console"]'
    "ray_kwargs.ray_init.num_cpus=$RAY_CPUS"
    "+ray_kwargs.ray_init.include_dashboard=false"
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONSTARTUP=$SLURM_SUBMIT_DIR/scripts/verl_vllm_compat.py"
    "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE=disabled"
)

echo "--- Hydra config preflight ---"
python3 -m verl.trainer.main_ppo "${GRPO_ARGS[@]}" --cfg job >/dev/null
echo "Hydra config OK"

TRAIN_LOG="logs/grpo_fullft_train_${SLURM_JOB_ID}.log"
echo "--- launching FULL-FT GRPO training (tee -> $TRAIN_LOG) ---"

python3 -c "exec(open('$SLURM_SUBMIT_DIR/scripts/verl_vllm_compat.py').read())"

(
    while sleep 300; do
        echo "=== HEARTBEAT $(date -Iseconds) job=${SLURM_JOB_ID} ===" | tee -a "$TRAIN_LOG"
        grep -iE "Training Progress|validation generation|step|epoch|global_step|ActorDied|Error|Traceback|OutOfMemory" \
            "logs/grpo_fullft_${SLURM_JOB_ID}.err" "$TRAIN_LOG" 2>/dev/null \
            | grep -v file_system_monitor | tail -5 | tee -a "$TRAIN_LOG" || true
        ls "$OUTPUT_DIR"/global_step_* 2>/dev/null | tail -2 | tee -a "$TRAIN_LOG" || true
    done
) &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT

set +e
stdbuf -oL -eL bash "$VERL_ENTRYPOINT" "${GRPO_ARGS[@]}" 2>&1 | tee -a "$TRAIN_LOG"
GRPO_EXIT=${PIPESTATUS[0]}
set -e
kill "$HEARTBEAT_PID" 2>/dev/null || true

if [[ "$GRPO_EXIT" -ne 0 ]]; then
    echo "ERROR: FULL-FT GRPO training exited with code $GRPO_EXIT" >&2
    exit "$GRPO_EXIT"
fi

echo "=== [$SLURM_JOB_ID] FULL-FT GRPO training finished at $(date) ==="
