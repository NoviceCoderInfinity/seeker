#!/usr/bin/env bash
set -euo pipefail

python3 -m vidcurator.run evaluate --config configs/kill_test.json
python3 -m unittest discover -s tests
