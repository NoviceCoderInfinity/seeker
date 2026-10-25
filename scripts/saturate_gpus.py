#!/usr/bin/env python3
"""Keep every SLURM partition topped up to its QOS ceiling from a priority backlog.

Idempotent: skips a backlog item if its job name is already running/pending or its
RUN_DIR/summary.json already exists. Submits highest-priority pending items into
whichever allowed partition has a free running slot. Safe to call repeatedly
(e.g. from a scheduled wakeup) — it only ever ADDS jobs, never cancels.

QOS running-slot ceilings (MaxJobsPU): dgx=4, l40=4, a40=3.
Qwen3 (8B) OOMs on 46GB cards → dgx only. Qwen2.5 (7B) runs anywhere with the pixel cap.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/vis-comp/22b3929/vidcurator")
JOB = "jobs/12_zero_shot_single.sh"
MODEL_Q25 = "models/qwen2_5_vl_7b"
MODEL_Q3 = "models/qwen3_vl_8b_instruct"
MODEL_IV = "models/internvl3_5_8b"

# Running-slot ceiling per partition (QOS MaxJobsPU).
CEIL = {"dgx": 4, "l40": 4, "a40": 3}
# Nodes with wedged GPUs (CUDA init fails/hangs — job dies ~1s exit 53 with no logs, or
# torch.cuda.is_available() hangs). cn11-dgx wedged 2026-07-04; cn14-dgx + cn15-dgx wedged
# 2026-07-06 (4 E-E jobs died instantly on them). Clear each once an admin resets the node.
BAD_NODES = "cn11-dgx,cn14-dgx,cn15-dgx"

# Priority backlog. Each item: name, config, qa, frames, model, allowed partitions, port.
# Order = priority (top first). Ports must be unique across concurrently-running jobs.
def _item(name, cfg, qa, frames, model, parts, port):
    return {"name": name, "config": cfg, "qa": qa, "frames": frames,
            "model": model, "parts": parts, "port": port,
            "run_dir": "runs/" + Path(cfg).stem}

BACKLOG = [
    # R5 Qwen3 pilots (dgx only) — confirms gains hold on the stronger backbone.
    _item("r5-nextgqa-q3", "configs/r5_v2_nextgqa_qwen3.json", "data/processed/nextgqa_val_150.jsonl", "data/indices/nextgqa_siglip_frames.jsonl", MODEL_Q3, ["dgx"], 8080),
    _item("r5-nextqa-q3", "configs/r5_v2_nextqa_qwen3.json", "data/processed/nextqa_temporal_val_150.jsonl", "data/indices/nextqa_temporal_siglip_frames.jsonl", MODEL_Q3, ["dgx"], 8081),
    _item("r5-lvb-q3", "configs/r5_v2_longvideobench_qwen3.json", "data/processed/longvideobench_val_150.jsonl", "data/indices/longvideobench_siglip_frames.jsonl", MODEL_Q3, ["dgx"], 8082),
    _item("r5-videomme-q3", "configs/r5_v2_videomme_qwen3.json", "data/processed/videomme_long_150.jsonl", "data/indices/videomme_long_siglip_frames.jsonl", MODEL_Q3, ["dgx"], 8083),
    _item("r5-mlvu-q3", "configs/r5_v2_mlvu_qwen3.json", "data/processed/mlvu_val_150.jsonl", "data/indices/mlvu_150_siglip_frames.jsonl", MODEL_Q3, ["dgx"], 8084),
    _item("r5-egoschema-q3", "configs/r5_v2_egoschema_qwen3.json", "data/processed/egoschema_val_150.jsonl", "data/indices/egoschema_siglip_frames.jsonl", MODEL_Q3, ["dgx"], 8085),
    # R2 full-val remaining (Qwen2.5, prefer a40/l40 to leave dgx for Qwen3).
    _item("r2-videomme-full", "configs/r2_v2_videomme_qwen2_5_full.json", "data/processed/videomme_long_half.jsonl", "data/indices/videomme_long_siglip_frames.jsonl", MODEL_Q25, ["a40", "l40"], 8090),
    _item("r2-lvb-full", "configs/r2_v2_longvideobench_qwen2_5_full.json", "data/processed/longvideobench_val_half.jsonl", "data/indices/longvideobench_siglip_frames.jsonl", MODEL_Q25, ["a40", "l40"], 8091),
    _item("r2-nextgqa-full", "configs/r2_v2_nextgqa_qwen2_5_full.json", "data/processed/nextgqa_val_half.jsonl", "data/indices/nextgqa_siglip_frames.jsonl", MODEL_Q25, ["a40", "l40"], 8092),
    _item("r2-nextqa-full", "configs/r2_v2_nextqa_qwen2_5_full.json", "data/processed/nextqa_temporal_test_3500_half.jsonl", "data/indices/nextqa_temporal_siglip_frames.jsonl", MODEL_Q25, ["a40", "l40"], 8093),
    # R6 InternVL3.5-8B cross-FAMILY backbone — LOWEST priority, idle l40/a40 only (never dgx →
    # keeps dgx free for GRPO/R5). 8B fits on 46GB at 448px 1-tile/frame (smoke-verified on L40S).
    _item("r6-nextgqa-iv", "configs/r6_v2_nextgqa_internvl.json", "data/processed/nextgqa_val_150.jsonl", "data/indices/nextgqa_siglip_frames.jsonl", MODEL_IV, ["l40", "a40"], 8100),
    _item("r6-nextqa-iv", "configs/r6_v2_nextqa_internvl.json", "data/processed/nextqa_temporal_val_150.jsonl", "data/indices/nextqa_temporal_siglip_frames.jsonl", MODEL_IV, ["l40", "a40"], 8101),
    # Long-video InternVL runs prefer dgx (80GB): l40/a40 44-48GB OOM at high frame budgets (r6-mlvu died).
    _item("r6-lvb-iv", "configs/r6_v2_longvideobench_internvl.json", "data/processed/longvideobench_val_150.jsonl", "data/indices/longvideobench_siglip_frames.jsonl", MODEL_IV, ["dgx", "l40", "a40"], 8102),
    _item("r6-videomme-iv", "configs/r6_v2_videomme_internvl.json", "data/processed/videomme_long_150.jsonl", "data/indices/videomme_long_siglip_frames.jsonl", MODEL_IV, ["dgx", "l40", "a40"], 8103),
    _item("r6-mlvu-iv", "configs/r6_v2_mlvu_internvl.json", "data/processed/mlvu_val_150.jsonl", "data/indices/mlvu_150_siglip_frames.jsonl", MODEL_IV, ["dgx", "l40", "a40"], 8104),
    _item("r6-egoschema-iv", "configs/r6_v2_egoschema_internvl.json", "data/processed/egoschema_val_150.jsonl", "data/indices/egoschema_siglip_frames.jsonl", MODEL_IV, ["l40", "a40"], 8105),
]


def squeue_names_and_counts():
    out = subprocess.run(
        ["squeue", "-u", "22b3929", "-h", "-o", "%j|%P|%t"],
        capture_output=True, text=True).stdout
    names, running = set(), {"dgx": 0, "l40": 0, "a40": 0}
    for line in out.splitlines():
        if not line.strip():
            continue
        name, part, st = (line.split("|") + ["", "", ""])[:3]
        names.add(name)
        if st == "R" and part in running:
            running[part] += 1
        elif st in ("PD", "CG") and part in running:
            running[part] += 1  # pending counts toward the slot we intend to fill
    return names, running


def submit(item, part):
    env = (f"ALL,CONFIG_PATH={item['config']},MODEL_PATH={item['model']},"
           f"QA_PATH={item['qa']},FRAMES_PATH={item['frames']},"
           f"RUN_DIR={item['run_dir']},PORT={item['port']},JOB_LABEL={item['name']}")
    cmd = ["sbatch", f"--partition={part}", f"--qos={part}", "--gres=gpu:1",
           f"--exclude={BAD_NODES}",
           f"--job-name=vc-{item['name']}", f"--export={env}", JOB]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return (r.stdout + r.stderr).strip()


def main():
    names, running = squeue_names_and_counts()
    print(f"running/pending per partition: {running}")
    submitted = 0
    for item in BACKLOG:
        jobname = f"vc-{item['name']}"
        if jobname in names:
            continue  # already queued/running
        if (ROOT / item["run_dir"] / "summary.json").exists():
            continue  # already finished
        # find an allowed partition with a free slot
        for part in item["parts"]:
            if running[part] < CEIL[part]:
                msg = submit(item, part)
                if "Submitted batch job" in msg:
                    running[part] += 1
                    submitted += 1
                    print(f"  + {jobname} -> {part}: {msg.splitlines()[-1]}")
                else:
                    print(f"  ! {jobname} -> {part} FAILED: {msg.splitlines()[-1]}")
                break
    print(f"saturator: submitted {submitted} job(s). "
          f"free slots left: " + ", ".join(f"{p}={CEIL[p]-running[p]}" for p in CEIL))
    return 0


if __name__ == "__main__":
    sys.exit(main())
