#!/usr/bin/env bash
# Setup vidcurator-grpo for GRPO (verl + trl + vllm). Does NOT modify vidcurator.
set -euo pipefail

ROOT="/home/vis-comp/22b3929/vidcurator"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="vidcurator-grpo"
SOURCE_ENV="vidcurator-train"
LOG="${ROOT}/env/setup_grpo_install.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== setup_grpo_env.sh started: $(date) ==="

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

echo "--- Step 1: verify ${SOURCE_ENV} ---"
conda activate "${SOURCE_ENV}"
python -c "import torch, transformers, peft; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('peft:', peft.__version__)"
conda deactivate

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "--- Step 2: ${ENV_NAME} already exists ---"
else
  echo "--- Step 2: cloning ${SOURCE_ENV} -> ${ENV_NAME} ---"
  conda create -n "${ENV_NAME}" --clone "${SOURCE_ENV}" -y
fi

echo "--- Step 3: install pinned GRPO stack ---"
conda activate "${ENV_NAME}"
cd "${ROOT}"

pip install -r env/grpo_requirements.txt
pip install 'vllm==0.11.2' --only-binary=:all:

# Prefer repo verl (matches third_party/verl entrypoint scripts).
pip install -e third_party/verl

echo "--- Step 4: import + compat verification ---"
export PYTHONSTARTUP="${ROOT}/scripts/verl_vllm_compat.py"
python scripts/verify_grpo_env.py --repo "${ROOT}" --skip-hydra

echo "VERL_ROOT=${ROOT}/third_party/verl"
echo "ENV_PATH=${CONDA_ROOT}/envs/${ENV_NAME}"
echo "=== finished: $(date) ==="
