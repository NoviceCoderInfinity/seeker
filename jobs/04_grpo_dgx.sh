#!/bin/bash
#SBATCH --job-name=vidcurator-grpo
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

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

# verl imports vLLM run_headless at module load; older vLLM wheels lack it.
export PYTHONSTARTUP="${SLURM_SUBMIT_DIR}/scripts/verl_vllm_compat.py"

# Cluster modules sometimes export ROCm vars even on NVIDIA nodes. verl/Ray
# rejects ROCR_VISIBLE_DEVICES when CUDA_VISIBLE_DEVICES is also set (143538).
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset PYTORCH_CUDA_ALLOC_CONF
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0

# Ray startup hygiene (job 143588: ray.init hung; 143770: socket path too long under repo).
export RAY_TMPDIR="/tmp/ray_${USER}_${SLURM_JOB_ID:-local}"
mkdir -p "$RAY_TMPDIR"
export RAY_DISABLE_IMPORT_WARNING=1
# `ray stop --force` kills EVERY ray process this user owns on the node — including a
# concurrent verl job's (149174/149175 mutually wedged on cn42-l40). Only clean up when
# we are the sole job of ours on this node.
OTHER_MY_JOBS=$(squeue -h -u "$USER" -w "$(hostname -s)" -t RUNNING -o "%i" 2>/dev/null | grep -cv "^${SLURM_JOB_ID}$" || true)
if [[ "${OTHER_MY_JOBS:-0}" -eq 0 ]]; then
    ray stop --force 2>/dev/null || true
else
    echo "Skipping 'ray stop --force': $OTHER_MY_JOBS other job(s) of $USER on $(hostname -s)"
fi

echo "GPU env after cleanup: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"

MODEL_INDEX="${MODEL_INDEX:-0}"
GRPO_CONFIG="${GRPO_CONFIG:-configs/grpo_stage1_pilot.json}"

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
    echo "Expected: \$REPO/third_party/verl/examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh" >&2
    exit 1
fi

echo "=== [$SLURM_JOB_ID] GRPO training started at $(date) ==="
echo "Model index: $MODEL_INDEX"
echo "Config: $GRPO_CONFIG"
echo "VERL_ROOT: $VERL_ROOT"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"

python -c "
import json
cfg = json.load(open('$GRPO_CONFIG'))
v = cfg['verl']
model = cfg['base_models'][int('$MODEL_INDEX')]
print(f'Base model: {model}')
print(f'LoRA rank: {v[\"lora_rank\"]}')
print(f'Global batch size: {v[\"global_batch_size\"]}')
print(f'Rollout n: {v[\"rollout_n\"]}')
print(f'Epochs: {v.get(\"epochs\", 3)}')
print(f'Train max samples: {v.get(\"train_max_samples\", \"all\")}')
print(f'Val max samples: {v.get(\"val_max_samples\", \"all\")}')
print(f'Rollout TP: {v.get(\"rollout_tp\", 2)}')
print(f'Budget curriculum: {cfg[\"budget_curriculum\"]}')
print('Reward weights:')
for k, val in cfg['reward_weights'].items():
    print(f'  {k}: {val}')
"

TRAIN_DATA="${TRAIN_DATA:-data/rl/vidcurator_train_sft.jsonl}"
EVAL_DATA="${EVAL_DATA:-data/rl/vidcurator_val_sft.jsonl}"

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "ERROR: training data not found: $TRAIN_DATA" >&2
    exit 1
fi

MODEL_PATH=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['base_models'][int('$MODEL_INDEX')])")
LORA_RANK=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['verl']['lora_rank'])")
LORA_ALPHA=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['verl'].get('lora_alpha', json.load(open('$GRPO_CONFIG'))['verl']['lora_rank']))")
GLOBAL_BS=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['verl']['global_batch_size'])")
ROLLOUT_N=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['verl']['rollout_n'])")
TRAIN_MAX=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['verl'].get('train_max_samples', -1))")
VAL_MAX=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['verl'].get('val_max_samples', -1))")
ROLLOUT_TP=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['verl'].get('rollout_tp', 2))")
SAVE_FREQ=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['verl'].get('save_freq', 20))")
TEST_FREQ=$(python -c "import json; print(json.load(open('$GRPO_CONFIG'))['verl'].get('test_freq', 20))")
OUTPUT_DIR="${GRPO_OUTPUT_DIR:-runs/grpo/$(basename "$MODEL_PATH")}"

# Reward regime for the outcome-aware GRPO study (default outcome; imitation = old baseline).
export VIDCURATOR_REWARD_MODE="${VIDCURATOR_REWARD_MODE:-outcome}"
export VIDCURATOR_GOLD_MAP="${VIDCURATOR_GOLD_MAP:-$SLURM_SUBMIT_DIR/data/rl/qid_gold.json}"
# Frame-penalty lambda knob (reward-lambda sweep): empty -> reward uses per-mode default
# (+0.2 coverage for outcome_eff, -0.2 localization for outcome_loc).
export VIDCURATOR_FRAME_PENALTY="${VIDCURATOR_FRAME_PENALTY:-}"
export VIDCURATOR_PROBE_SHAPE="${VIDCURATOR_PROBE_SHAPE:-}"
echo "Reward mode: $VIDCURATOR_REWARD_MODE  frame_penalty: '${VIDCURATOR_FRAME_PENALTY}' probe_shape: '${VIDCURATOR_PROBE_SHAPE}'  gold_map: $VIDCURATOR_GOLD_MAP  output_dir: $OUTPUT_DIR"

# Separate parquet dirs let concurrent runs train on different data without clobbering
# each other (probe-sweep jobs share data/rl/parquet; N5 clean retrain uses its own).
PARQUET_DIR="${PARQUET_DIR:-data/rl/parquet}"
if [[ "${GRPO_SKIP_PREP:-0}" == "1" && -f "$PARQUET_DIR/train.parquet" ]]; then
    echo "--- GRPO_SKIP_PREP=1: reusing existing parquet in $PARQUET_DIR ---"
else
    echo "--- preparing verl parquet -> $PARQUET_DIR ---"
    python scripts/prep_grpo_parquet.py \
        --train "$TRAIN_DATA" \
        --val "$EVAL_DATA" \
        --out-dir "$PARQUET_DIR"
fi

TRAIN_PARQUET="$SLURM_SUBMIT_DIR/$PARQUET_DIR/train.parquet"
EVAL_PARQUET="$SLURM_SUBMIT_DIR/$PARQUET_DIR/val.parquet"

echo "--- rendering verl GRPO command ---"
python scripts/render_verl_grpo_command.py \
    --config "$GRPO_CONFIG" \
    --model-index "$MODEL_INDEX"

VERL_ENTRYPOINT="$VERL_ROOT/examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh"

if [[ ! -f "$VERL_ENTRYPOINT" ]]; then
    echo "ERROR: verl entrypoint not found: $VERL_ENTRYPOINT" >&2
    exit 1
fi

NGPUS="${SLURM_GPUS_ON_NODE:-4}"
RAY_CPUS="${SLURM_CPUS_PER_TASK:-16}"
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
export EXPERIMENT_NAME="${GRPO_EXPERIMENT:-stage1_$(basename "$MODEL_PATH")}"
export MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT:-2048}"
export MAX_RESPONSE_LENGTH="${GRPO_MAX_RESPONSE:-256}"

SFT_ADAPTER="${SFT_ADAPTER:-runs/sft/qwen2_5_vl_7b_curator_lora}"

GRPO_ARGS=(
    data.train_files="$TRAIN_PARQUET"
    data.val_files="$EVAL_PARQUET"
    data.prompt_key=prompt
    "data.train_max_samples=$TRAIN_MAX"
    "data.val_max_samples=$VAL_MAX"
    "data.val_batch_size=32"
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.lora_rank="$LORA_RANK"
    "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA"
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.use_remove_padding=False
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
    trainer.default_local_dir="$OUTPUT_DIR"
    data.train_batch_size="$GLOBAL_BS"
    actor_rollout_ref.rollout.n="$ROLLOUT_N"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.free_cache_engine=False
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GRPO_GPU_MEM_UTIL:-0.55}"
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
    actor_rollout_ref.model.target_modules=all-linear
    "actor_rollout_ref.model.exclude_modules='.*visual.*'"
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
    "+ray_kwargs.ray_init.runtime_env.env_vars.VIDCURATOR_REWARD_MODE='$VIDCURATOR_REWARD_MODE'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VIDCURATOR_GOLD_MAP='$VIDCURATOR_GOLD_MAP'"
    # NOTE: inner quotes force Hydra to treat the value as a string — Ray rejects non-str
    # env_vars (job 148679 died: "-0.2 is of type float").
    "+ray_kwargs.ray_init.runtime_env.env_vars.VIDCURATOR_FRAME_PENALTY='$VIDCURATOR_FRAME_PENALTY'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VIDCURATOR_PROBE_SHAPE='$VIDCURATOR_PROBE_SHAPE'"
)

if [[ -f "$SFT_ADAPTER/adapter_config.json" ]]; then
    echo "Loading SFT cold-start adapter: $SFT_ADAPTER"
    GRPO_ARGS+=( "actor_rollout_ref.model.lora_adapter_path=$SFT_ADAPTER" )
else
    echo "WARN: SFT adapter not found at $SFT_ADAPTER; training LoRA from scratch"
fi

echo "--- Hydra config preflight ---"
python3 -m verl.trainer.main_ppo "${GRPO_ARGS[@]}" --cfg job >/dev/null
echo "Hydra config OK"

TRAIN_LOG="logs/grpo_train_${SLURM_JOB_ID}.log"
echo "--- launching GRPO training (tee -> $TRAIN_LOG) ---"
echo "Expected steps: ~$(( (TRAIN_MAX > 0 ? TRAIN_MAX : 18909) / GLOBAL_BS * TOTAL_EPOCHS )) optimizer steps" | tee -a "$TRAIN_LOG"

# Ensure flash_attn / vLLM compat patches load in driver + Ray workers.
python3 -c "exec(open('$SLURM_SUBMIT_DIR/scripts/verl_vllm_compat.py').read())"

(
    while sleep 300; do
        # stdout only — teeing grep hits back into TRAIN_LOG makes the next grep re-match
        # its own output (149174's log filled with recursive filename prefixes).
        echo "=== HEARTBEAT $(date -Iseconds) job=${SLURM_JOB_ID} ==="
        grep -iE "Training Progress|validation generation|ActorDied|Error|Traceback" \
            "logs/grpo_stage1_${SLURM_JOB_ID}.err" "$TRAIN_LOG" 2>/dev/null \
            | grep -v "file_system_monitor\|HEARTBEAT" | tail -3 || true
        ls -d "$OUTPUT_DIR"/global_step_* 2>/dev/null | tail -2 || true
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
    echo "ERROR: GRPO training exited with code $GRPO_EXIT" >&2
    exit "$GRPO_EXIT"
fi

echo "=== [$SLURM_JOB_ID] GRPO training finished at $(date) ==="
