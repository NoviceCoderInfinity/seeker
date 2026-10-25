#!/usr/bin/env python3
"""Fail-fast preflight for GRPO jobs before allocating DGX GPUs."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"PREFLIGHT FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def ok(msg: str) -> None:
    print(f"PREFLIGHT OK: {msg}")


def check_python_packages() -> None:
    for name in ("verl", "vllm", "peft", "trl", "ray", "pandas", "pyarrow"):
        try:
            mod = importlib.import_module(name)
        except ImportError as exc:
            fail(f"missing package {name}: {exc}")
        version = getattr(mod, "__version__", "?")
        ok(f"{name} {version}")


def check_run_headless() -> None:
    # Compat shim should have run before this script if PYTHONSTARTUP is set.
    try:
        from vllm.entrypoints.cli.serve import run_headless  # noqa: F401
    except ImportError as exc:
        fail(f"vLLM run_headless still missing after compat shim: {exc}")
    ok("vLLM run_headless import")


def check_gpu_env() -> None:
    rocr = os.environ.get("ROCR_VISIBLE_DEVICES")
    hip = os.environ.get("HIP_VISIBLE_DEVICES")
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    if rocr and cuda:
        fail(
            "ROCR_VISIBLE_DEVICES is set together with CUDA_VISIBLE_DEVICES; "
            "unset ROCR/HIP before launching verl (see jobs/04_grpo_dgx.sh)."
        )
    if rocr or hip:
        fail(f"ROCR/HIP env vars still set: ROCR={rocr!r} HIP={hip!r}")
    ok(f"GPU env clean (CUDA_VISIBLE_DEVICES={cuda or 'unset'})")


def check_paths(repo: Path, config_path: Path, model_index: int) -> None:
    verl_entry = repo / "third_party/verl/examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh"
    if not verl_entry.is_file():
        fail(f"missing verl entrypoint: {verl_entry}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    model = cfg["base_models"][model_index]
    if not (repo / model / "config.json").is_file():
        fail(f"base model not found: {repo / model}")

    train = repo / cfg["train_data"]
    val = repo / cfg["eval_data"]
    if not train.is_file():
        fail(f"train data missing: {train}")
    if not val.is_file():
        fail(f"eval data missing: {val}")

    adapter = repo / "runs/sft/qwen2_5_vl_7b_curator_lora/adapter_config.json"
    if adapter.is_file():
        ok(f"SFT adapter present: {adapter.parent}")
    else:
        print("PREFLIGHT WARN: SFT adapter missing; GRPO will train LoRA from scratch")

    ok(f"config/model/data paths for {model}")


def check_hydra(repo: Path, config_path: Path, model_index: int) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    model = cfg["base_models"][model_index]
    verl_cfg = cfg["verl"]
    train_parquet = repo / "data/rl/parquet/train.parquet"
    val_parquet = repo / "data/rl/parquet/val.parquet"
    if not train_parquet.is_file() or not val_parquet.is_file():
        subprocess.run(
            [
                sys.executable,
                str(repo / "scripts/prep_grpo_parquet.py"),
                "--train",
                str(repo / cfg["train_data"]),
                "--val",
                str(repo / cfg["eval_data"]),
                "--out-dir",
                str(repo / "data/rl/parquet"),
            ],
            check=True,
            cwd=repo,
        )

    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"data.train_files={train_parquet}",
        f"data.val_files={val_parquet}",
        "data.prompt_key=prompt",
        f"actor_rollout_ref.model.path={model}",
        f"actor_rollout_ref.model.lora_rank={verl_cfg['lora_rank']}",
        "actor_rollout_ref.model.use_fused_kernels=False",
        "actor_rollout_ref.model.use_remove_padding=False",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        f"trainer.default_local_dir={repo / 'runs/grpo' / Path(model).name}",
        f"data.train_batch_size={verl_cfg['global_batch_size']}",
        f"actor_rollout_ref.rollout.n={verl_cfg['rollout_n']}",
        "trainer.n_gpus_per_node=4",
        'trainer.logger=["console"]',
        "--cfg",
        "job",
    ]
    adapter = repo / "runs/sft/qwen2_5_vl_7b_curator_lora"
    if (adapter / "adapter_config.json").is_file():
        cmd.insert(-2, f"actor_rollout_ref.model.lora_adapter_path={adapter}")

    subprocess.run(cmd, check=True, cwd=repo, stdout=subprocess.DEVNULL)
    ok("Hydra GRPO config composes")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--config", default="configs/grpo_stage1.json")
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--skip-hydra", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    config_path = (repo / args.config).resolve()

    # Ensure compat shim is active even when this script is invoked directly.
    compat = repo / "scripts/verl_vllm_compat.py"
    if compat.is_file():
        exec(compat.read_text(encoding="utf-8"), {"__name__": "__verl_vllm_compat__"})

    check_gpu_env()
    check_python_packages()
    check_run_headless()
    check_paths(repo, config_path, args.model_index)
    if not args.skip_hydra:
        check_hydra(repo, config_path, args.model_index)

    print("=== GRPO preflight passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
