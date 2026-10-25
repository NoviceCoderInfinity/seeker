#!/usr/bin/env bash
set -euo pipefail

python_bin="${1:-.conda-env/bin/python}"

"$python_bin" -m pip install --upgrade pip
"$python_bin" -m pip install -r env/requirements-zero-shot.txt
"$python_bin" -m unittest discover -s tests
