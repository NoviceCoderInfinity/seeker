#!/usr/bin/env bash
set -euo pipefail

cat <<'NOTE'
This script intentionally does not auto-download model weights.

Reason:
- Model pulls are large.
- Some models require license or gated-access checks.
- The current workspace has restricted network access.

After approval and Hugging Face auth, use commands like:

  huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir models/qwen2_5_vl_7b \
    --local-dir-use-symlinks False

  huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir models/qwen3_vl_8b_instruct \
    --local-dir-use-symlinks False

Then run:

  python3 scripts/check_research_readiness.py
NOTE
