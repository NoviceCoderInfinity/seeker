#!/usr/bin/env python3
"""Generate learned-GRPO eval configs from Qwen2.5 zero-shot configs."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ZS_GLOB = "configs/gpu_*_qwen2_5_zero_shot.json"
ADAPTER = "runs/grpo/qwen2_5_vl_7b_grpo_lora/lora_adapter"
MODEL = "models/qwen2_5_vl_7b"


def main() -> int:
    written = 0
    for zs_path in sorted(REPO.glob(ZS_GLOB)):
        zs = json.loads(zs_path.read_text())
        stem = zs_path.stem.replace("_zero_shot", "_learned_grpo")
        out_cfg = REPO / "configs" / f"{stem}.json"
        run_dir = f"runs/{stem}"
        cfg = {
            "name": stem,
            "frames_path": zs["frames_path"],
            "qa_path": zs["qa_path"],
            "output_dir": run_dir,
            "budgets": zs["budgets"],
            "policies": ["scripted_curator", "learned_curator"],
            "action_policy": {
                "model_path": MODEL,
                "adapter_path": ADAPTER,
                "max_new_tokens": 384,
            },
            "answerer": {
                "kind": "openai_compat",
                "endpoint": "http://localhost:8000",
                "model": MODEL,
                "send_images": True,
                "max_images": 16,
                "timeout_seconds": 600,
            },
            "policy": zs.get("policy", {}),
            "retrieval_recall_at_k": zs.get("retrieval_recall_at_k", [4, 8, 16, 32]),
        }
        out_cfg.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_cfg.relative_to(REPO)}")
        written += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
