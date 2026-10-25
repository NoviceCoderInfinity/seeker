#!/usr/bin/env python3
"""Render a verl command skeleton from the GRPO config."""

from __future__ import annotations

import argparse
import json
import os


def expand(value: str) -> str:
    return os.path.expandvars(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-index", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    model = cfg["base_models"][args.model_index]
    verl = cfg["verl"]
    repo = expand(verl["repo"])
    entrypoint = os.path.join(repo, verl["entrypoint"])
    command = [
        "bash",
        entrypoint,
        f"model.path={model}",
        f"data.train_files={expand(cfg['train_data'])}",
        f"data.val_files={expand(cfg['eval_data'])}",
        f"trainer.default_local_dir={expand(cfg['output_root'])}/{model.split('/')[-1]}",
        f"actor_rollout_ref.model.lora_rank={verl['lora_rank']}",
        f"data.train_batch_size={verl['global_batch_size']}",
        f"actor_rollout_ref.rollout.n={verl['rollout_n']}",
    ]
    print(" ".join(command))
    print()
    print("Before running, wire VID-CURATOR tool actions and reward components into the verl rollout worker.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
