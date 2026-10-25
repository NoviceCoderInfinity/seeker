#!/usr/bin/env python3
"""Check local readiness for VID-CURATOR experiments."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def command_status(command: str) -> dict:
    exe = shutil.which(command)
    local_exe = Path.cwd() / ".conda-env" / "bin" / command
    if exe is None and local_exe.exists():
        exe = str(local_exe)
    row = {"command": command, "available": exe is not None, "path": exe}
    if exe:
        try:
            proc = subprocess.run([str(exe), "--version"], text=True, capture_output=True, timeout=10)
            row["version_head"] = (proc.stdout or proc.stderr).splitlines()[:2]
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            row["version_error"] = str(exc)
    return row


def file_status(path: str) -> dict:
    p = Path(path)
    return {"path": path, "exists": p.exists(), "is_file": p.is_file(), "is_dir": p.is_dir()}


def jsonl_count(path: str) -> int | None:
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def glob_count(pattern: str) -> int:
    return sum(1 for _ in Path.cwd().glob(pattern))


def import_status(module: str) -> dict:
    try:
        imported = importlib.import_module(module)
    except Exception as exc:  # pragma: no cover - environment diagnostic
        return {"module": module, "available": False, "error": repr(exc)}
    return {"module": module, "available": True, "version": getattr(imported, "__version__", "")}


def torch_status() -> dict:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment diagnostic
        return {"available": False, "error": repr(exc)}
    cuda = torch.cuda.is_available()
    devices = []
    if cuda:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gib": round(props.total_memory / (1024**3), 2),
                }
            )
    return {
        "available": True,
        "torch_version": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if cuda else 0,
        "devices": devices,
    }


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/readiness_report.json")
    args = parser.parse_args()

    root = Path.cwd()
    report = {
        "root": str(root),
        "python": sys.version,
        "commands": {
            name: command_status(name)
            for name in ["python3", "ffmpeg", "ffprobe", "nvidia-smi", "git", "git-lfs", "conda"]
        },
        "python_imports": {
            name: import_status(name)
            for name in [
                "torch",
                "transformers",
                "accelerate",
                "vllm",
                "sglang",
                "datasets",
                "pandas",
                "pyarrow",
                "PIL",
                "qwen_vl_utils",
            ]
        },
        "torch": torch_status(),
        "workspace_files": {
            path: file_status(path)
            for path in [
                "plan.md",
                "README.md",
                "pyproject.toml",
                "configs/kill_test.json",
                "configs/zero_shot_matrix.json",
                "manifests/datasets.json",
                "manifests/models.json",
                "env/environment.h100.yml",
            ]
        },
        "data_dirs": {
            path: file_status(path)
            for path in [
                "data/raw",
                "data/processed",
                "data/videos",
                "data/frames",
                "data/indices",
                "models",
                "runs",
            ]
        },
        "datasets": [],
        "models": [],
        "asset_inventory": {
            "processed_jsonl_rows": {
                path: jsonl_count(path)
                for path in [
                    "data/processed/nextgqa_val.jsonl",
                    "data/processed/nextgqa_val_150.jsonl",
                    "data/processed/videomme_all.jsonl",
                    "data/processed/videomme_long.jsonl",
                    "data/processed/videomme_long_150.jsonl",
                    "data/processed/egoschema_val.jsonl",
                    "data/processed/egoschema_val_150.jsonl",
                    "data/processed/egoschema_test_hidden.jsonl",
                    "data/processed/nextqa_temporal_val.jsonl",
                    "data/processed/nextqa_temporal_val_150.jsonl",
                ]
            },
            "raw_video_counts": {
                "nextgqa_mp4": glob_count("data/raw/nextgqa/NExTVideo/**/*.mp4"),
                "videomme_zip_chunks": glob_count("data/raw/videomme/videos_chunked_*.zip"),
                "egoschema_zip_chunks": glob_count("data/raw/egoschema/videos_chunked_*.zip"),
                "videomme_extracted_videos": glob_count("data/videos/videomme/**/*.mp4"),
                "egoschema_extracted_videos": glob_count("data/videos/egoschema/**/*.mp4"),
                "nextqa_extracted_videos": glob_count("data/videos/nextqa/**/*.mp4"),
            },
            "frame_index_rows": {
                path: jsonl_count(path)
                for path in [
                    "data/indices/nextgqa_timestamp_frames.jsonl",
                    "data/indices/nextgqa_val_150_decoded_frames.jsonl",
                    "data/indices/nextgqa_siglip_frames.jsonl",
                    "data/indices/videomme_long_siglip_frames.jsonl",
                    "data/indices/egoschema_siglip_frames.jsonl",
                    "data/indices/nextqa_temporal_siglip_frames.jsonl",
                ]
            },
        },
    }

    datasets_path = root / "manifests/datasets.json"
    if datasets_path.exists():
        for item in load_json(datasets_path).get("datasets", []):
            report["datasets"].append(
                {
                    "name": item["name"],
                    "priority": item["priority"],
                    "processed": file_status(item["processed"]),
                    "small_split": file_status(item["small_split"]),
                    "raw_dir": file_status(item["raw_dir"]),
                    "needs_license_check": item.get("requires_license_check", False),
                }
            )
    models_path = root / "manifests/models.json"
    if models_path.exists():
        for item in load_json(models_path).get("models", []):
            local_dir = Path("models") / item["name"]
            report["models"].append(
                {
                    "name": item["name"],
                    "hf_id": item["hf_id"],
                    "priority": item["priority"],
                    "license": item["license"],
                    "local_dir": file_status(str(local_dir)),
                    "safetensors_files": glob_count(f"models/{item['name']}/**/*.safetensors"),
                    "bin_files": glob_count(f"models/{item['name']}/**/*.bin"),
                }
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {out}")
    missing_data = [d["name"] for d in report["datasets"] if not d["small_split"]["exists"]]
    missing_models = [m["name"] for m in report["models"] if not m["local_dir"]["exists"]]
    print(f"missing dataset splits: {', '.join(missing_data) if missing_data else 'none'}")
    print(f"missing local model dirs: {', '.join(missing_models) if missing_models else 'none'}")
    print(f"torch cuda available: {report['torch'].get('cuda_available')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
