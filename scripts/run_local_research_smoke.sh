#!/usr/bin/env bash
set -euo pipefail

python_bin="${1:-.conda-env/bin/python}"

"$python_bin" -m unittest discover -s tests
"$python_bin" scripts/check_research_readiness.py
"$python_bin" scripts/run_zero_shot_matrix.py \
  --matrix configs/zero_shot_matrix.json \
  --only-dataset fixture \
  --only-answerer lexical_fixture \
  --allow-missing
"$python_bin" scripts/export_paper_artifacts.py \
  --run-dir runs/zero_shot/fixture__lexical_fixture
"$python_bin" scripts/export_qualitative_traces.py \
  --run-dir runs/zero_shot/fixture__lexical_fixture \
  --qa-path data/fixtures/qa.jsonl \
  --frames-path data/fixtures/frames.jsonl \
  --policy scripted_curator \
  --max-examples 3

if [[ -f data/processed/nextgqa_val_150.jsonl && -f data/indices/nextgqa_timestamp_frames.jsonl ]]; then
  "$python_bin" scripts/run_zero_shot_matrix.py \
    --matrix configs/zero_shot_matrix.json \
    --only-dataset nextgqa_timestamp_150 \
    --only-answerer lexical_metadata \
    --allow-missing
  "$python_bin" scripts/export_paper_artifacts.py \
    --run-dir runs/zero_shot/nextgqa_timestamp_150__lexical_metadata
  "$python_bin" scripts/export_qualitative_traces.py \
    --run-dir runs/zero_shot/nextgqa_timestamp_150__lexical_metadata \
    --qa-path data/processed/nextgqa_val_150.jsonl \
    --frames-path data/indices/nextgqa_timestamp_frames.jsonl \
    --policy scripted_curator \
    --max-examples 5
fi
