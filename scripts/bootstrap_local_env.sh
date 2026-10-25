#!/usr/bin/env bash
set -euo pipefail

env_dir="${1:-.venv}"

if python3 -m venv "$env_dir"; then
  python_bin="$env_dir/bin/python"
else
  conda_bin="${CONDA_BIN:-/home/maram/anaconda3/condabin/conda}"
  conda_env="${CONDA_ENV_PATH:-.conda-env}"
  pkgs_dir="${CONDA_PKGS_DIRS:-/data/maram/project/.conda-pkgs}"
  CONDA_PKGS_DIRS="$pkgs_dir" "$conda_bin" create -p "$conda_env" python=3.10 pip -y
  python_bin="$conda_env/bin/python"
fi

"$python_bin" -m pip --version
"$python_bin" -m pip install --no-build-isolation -e .
"$python_bin" -m unittest discover -s tests

cat <<EOF
Local environment ready:
  source ${env_dir}/bin/activate  # if venv succeeded
  conda activate /data/maram/project/.conda-env  # if conda fallback was used
  python -m vidcurator.run evaluate --config configs/kill_test.json
EOF
