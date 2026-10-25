#!/usr/bin/env python3
"""Generate full-val ("full-scale") eval configs for the parallel-node run (plan5.md).

Emits declarative JSON configs that mirror the existing r1_v2/r2_v2 shape, but pointed at the
FULL validation split of each dataset (not the n=150 sample) and with `fs_`-prefixed names/run-dirs
so artifacts can never collide with the primary user's `r1_v2_*`/`r2_v2_*` runs.

Usage (run from the repo root of the SECOND user's copy):
  python scripts/gen_fullscale_configs.py --wave headline   --model qwen2_5_vl_7b
  python scripts/gen_fullscale_configs.py --wave selectors  --model qwen2_5_vl_7b
  python scripts/gen_fullscale_configs.py --wave ablations   --model qwen2_5_vl_7b
  python scripts/gen_fullscale_configs.py --wave bsf         --model qwen2_5_vl_7b
  # cross-family (plan5 §5): trim budgets to keep cost down; keep the set constant per family
  python scripts/gen_fullscale_configs.py --wave headline   --model internvl3_5_8b --budgets 4,8,16
  python scripts/gen_fullscale_configs.py --wave selectors  --model internvl3_5_8b --budgets 4,8,16
  # learned curator (plan5 §4.3) — run via jobs/26 (2 GPU), NOT jobs/50/12:
  python scripts/gen_fullscale_configs.py --wave learned --model qwen2_5_vl_7b \
      --adapter runs/grpo_fs/outcome_eff_r32/lora_adapter

Writes one config per dataset to configs/fs_<ds>_<model>_<wave>.json and prints the paths.
Idempotent: re-running overwrites the config text (never touches runs/).
"""
import argparse
import json
from pathlib import Path

# dataset -> (full-val QA path, full SigLIP index). FULL splits, not *_150 / *_450.
DATASETS = {
    "nextgqa":        ("data/processed/nextgqa_val.jsonl",          "data/indices/nextgqa_siglip_frames.jsonl"),
    "nextqa":         ("data/processed/nextqa_temporal_val.jsonl",  "data/indices/nextqa_temporal_siglip_frames.jsonl"),
    "videomme":       ("data/processed/videomme_long.jsonl",        "data/indices/videomme_long_siglip_frames.jsonl"),
    "egoschema":      ("data/processed/egoschema_val.jsonl",        "data/indices/egoschema_siglip_frames.jsonl"),
    "mlvu":           ("data/processed/mlvu_val.jsonl",             "data/indices/mlvu_siglip_frames.jsonl"),
    "longvideobench": ("data/processed/longvideobench_val.jsonl",   "data/indices/longvideobench_siglip_frames.jsonl"),
}

MODELS = {
    "qwen2_5_vl_7b":        "models/qwen2_5_vl_7b",
    "qwen3_vl_8b_instruct": "models/qwen3_vl_8b_instruct",
    "internvl3_5_8b":       "models/internvl3_5_8b",
    "llava_video_7b_qwen2": "models/llava_video_7b_qwen2",
}

# wave -> (policies, default budgets). Non-overlapping across headline/selectors/ablations so the
# union == the 12 R1 policies with no duplicated compute. bsf + learned are special (see below).
WAVES = {
    "headline":  (["uniform", "mmr_retrieve", "scripted_curator", "curator_siglip", "curator_probe_off"], [2, 4, 8, 16, 32]),
    "selectors": (["aks", "videotree", "flexframe"], [2, 4, 8, 16, 32]),
    "ablations": (["curator_curation_off", "curator_summary_off", "curator_sufficiency_off", "curator_retrieval_off"], [4, 8, 16]),
    "bsf":       (["best_single_frame"], [1]),
    "learned":   (["learned_curator"], [2, 4, 8, 16, 32]),
}

POLICY_INTERNALS = {"max_hops": 4, "retrieve_per_hop": 6, "seed_uniform": 2, "sufficiency_threshold": 0.72}


def build(ds, model, wave, budgets, adapter):
    qa, frames = DATASETS[ds]
    policies, _ = WAVES[wave]
    name = f"fs_{ds}_{model}_{wave}"
    cfg = {
        "name": name,
        "frames_path": frames,
        "qa_path": qa,
        "output_dir": f"runs/{name}",
        "budgets": budgets,
        "policies": policies,
        "answerer": {
            "kind": "openai_compat",
            "endpoint": "http://localhost:8000",
            "model": MODELS[model],
            "send_images": True,
            "max_images": 32,
            "timeout_seconds": 600,
        },
        "policy": dict(POLICY_INTERNALS),
        "retrieval_recall_at_k": [4, 8, 16, 32],
    }
    if wave == "learned":
        if not adapter:
            raise SystemExit("--adapter is required for --wave learned (path to the LoRA adapter dir)")
        cfg["action_policy"] = {"model_path": MODELS[model], "adapter_path": adapter, "max_new_tokens": 384}
        cfg["answerer"]["max_images"] = 16  # matches the primary learned configs
    return name, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", required=True, choices=list(WAVES))
    ap.add_argument("--model", default="qwen2_5_vl_7b", choices=list(MODELS))
    ap.add_argument("--budgets", default="", help="comma list to override the wave default, e.g. 4,8,16")
    ap.add_argument("--adapter", default="", help="LoRA adapter dir (required for --wave learned)")
    ap.add_argument("--datasets", default="", help="comma list to restrict; default = all 6")
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")] if args.budgets else WAVES[args.wave][1]
    dss = args.datasets.split(",") if args.datasets else list(DATASETS)
    Path("configs").mkdir(exist_ok=True)
    for ds in dss:
        name, cfg = build(ds, args.model, args.wave, budgets, args.adapter)
        out = Path("configs") / f"{name}.json"
        out.write_text(json.dumps(cfg, indent=2))
        print(out)


if __name__ == "__main__":
    main()
