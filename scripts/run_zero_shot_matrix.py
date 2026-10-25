#!/usr/bin/env python3
"""Run all available jobs in a zero-shot matrix config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vidcurator.evaluate import evaluate_config
from vidcurator.io import ensure_dir, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/zero_shot_matrix.json")
    parser.add_argument("--only-dataset", default="")
    parser.add_argument("--only-answerer", default="")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    with open(args.matrix, "r", encoding="utf-8") as handle:
        matrix = json.load(handle)

    output_root = Path(matrix.get("output_root", "runs/zero_shot"))
    ensure_dir(output_root)
    jobs = []
    for dataset in matrix["datasets"]:
        if args.only_dataset and dataset["name"] != args.only_dataset:
            continue
        for answerer in matrix["answerers"]:
            if args.only_answerer and answerer["name"] != args.only_answerer:
                continue
            if dataset["name"] not in answerer.get("datasets", []):
                continue
            jobs.append((dataset, answerer))

    manifest = {"jobs": [], "skipped": []}
    for dataset, answerer in jobs:
        frames_path = Path(dataset["frames_path"])
        qa_path = Path(dataset["qa_path"])
        job_name = f"{dataset['name']}__{answerer['name']}"
        if not frames_path.exists() or not qa_path.exists():
            row = {
                "job": job_name,
                "reason": "missing input",
                "frames_exists": frames_path.exists(),
                "qa_exists": qa_path.exists(),
            }
            manifest["skipped"].append(row)
            print(f"skip {job_name}: {row}")
            if not args.allow_missing:
                write_json(output_root / "matrix_manifest.json", manifest)
                return 1
            continue
        run_dir = output_root / job_name
        cfg = {
            "name": job_name,
            "frames_path": str(frames_path),
            "qa_path": str(qa_path),
            "output_dir": str(run_dir),
            "answerer": {k: v for k, v in answerer.items() if k not in {"name", "datasets"}},
            "budgets": matrix["budgets"],
            "policies": matrix["policies"],
            "policy": matrix.get("policy", {}),
            "retrieval_recall_at_k": matrix.get("retrieval_recall_at_k", [4, 8, 16]),
        }
        write_json(run_dir / "config.json", cfg)
        summary = evaluate_config(cfg)
        manifest["jobs"].append({"job": job_name, "run_dir": str(run_dir), "groups": len(summary["groups"])})
        print(f"done {job_name}: {run_dir}")

    write_json(output_root / "matrix_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
