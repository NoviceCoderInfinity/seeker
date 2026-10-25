#!/usr/bin/env python3
"""Significance pack: one command -> every paired comparison in the paper, from run dirs.

Usage:
  python scripts/sigpack.py --out reports/sigpack_v3.md \
      --cell nextgqa runs/p0_v3control_nextgqa data/processed/heldout/nextgqa_heldout.jsonl \
      ...
or with --auto (default): uses the built-in registry of paper cells below.

For each (dataset, run_dir) it reads _shards/*.jsonl, scores against gold, and emits
per-policy accuracy plus paired McNemar (exact binomial on discordant pairs) for every
policy-vs-uniform comparison at each budget, with Wilson 95% CIs.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

# ---- registry of paper cells: (label, run_dir, gold_qa_path) -------------------------
REGISTRY = [
    # v3 heldout money table
    ("nextgqa-ho", "runs/p0_v3control_nextgqa", "data/processed/heldout/nextgqa_heldout.jsonl"),
    ("nextgqa-ho", "runs/p0_v3control_nextgqa_scripted", "data/processed/heldout/nextgqa_heldout.jsonl"),
    ("nextqa-ho", "runs/v3re_nextqa", "data/processed/heldout/nextqa_heldout.jsonl"),
    ("egoschema-ho", "runs/v3re_egoschema", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("videomme-ho", "runs/v3re_videomme", "data/processed/heldout/videomme_heldout.jsonl"),
    ("mlvu-ho", "runs/v3re_mlvu", "data/processed/heldout/mlvu_heldout.jsonl"),
    ("lvbench-ho", "runs/v3re_lvbench", "data/processed/heldout/lvbench_heldout.jsonl"),
    # selectors
    ("nextgqa-ho", "runs/v3sel_nextgqa", "data/processed/heldout/nextgqa_heldout.jsonl"),
    ("nextqa-ho", "runs/v3sel_nextqa", "data/processed/heldout/nextqa_heldout.jsonl"),
    ("egoschema-ho", "runs/v3sel_egoschema", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("mlvu-ho", "runs/v3sel_mlvu", "data/processed/heldout/mlvu_heldout.jsonl"),
    # blind floors
    ("nextgqa-ho", "runs/blind_nextgqa", "data/processed/heldout/nextgqa_heldout.jsonl"),
    ("nextqa-ho", "runs/blind_nextqa", "data/processed/heldout/nextqa_heldout.jsonl"),
    ("egoschema-ho", "runs/blind_egoschema", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("mlvu-ho", "runs/blind_mlvu", "data/processed/heldout/mlvu_heldout.jsonl"),
    ("lvbench-ho", "runs/blind_lvbench", "data/processed/heldout/lvbench_heldout.jsonl"),
    # full-set training-free wave
    ("egoschema-full", "runs/full_v3_egoschema", "data/processed/egoschema_val.jsonl"),
    ("videomme-full", "runs/full_v3_videomme", "data/processed/videomme_long.jsonl"),
    ("videomme-full", "runs/full_v3_videomme_subs", "data/processed/videomme_long.jsonl"),
    ("nextgqa-full", "runs/full_v3_nextgqa_half", "data/processed/nextgqa_val_half.jsonl"),
    ("lvbench-full", "runs/full_v3_lvbench", "data/processed/lvbench_val.jsonl"),
    ("nextqa-full", "runs/full_v3_nextqa_half", "data/processed/nextqa_temporal_test_3500_half.jsonl"),
    # learned rows (firewall heldout ONLY)
    ("nextgqa-ho", "runs/n5_s100_nextgqa_heldout", "data/processed/heldout/nextgqa_heldout.jsonl"),
    ("nextgqa-ho", "runs/n5_s100_nextgqa_heldout_v3", "data/processed/heldout/nextgqa_heldout.jsonl"),
    # 2026-07-13 wave: mlvu full set, firewall-clean high-n learned rows, -scan ablation
    ("mlvu-full", "runs/full_v3_mlvu", "data/processed/mlvu_val.jsonl"),
    ("mlvu-full", "runs/full_v3_mlvu_b16", "data/processed/mlvu_val.jsonl"),
    ("nextgqa-cleaneval", "runs/full_v3_nextgqa_cleaneval", "data/processed/cleaneval/nextgqa_cleaneval.jsonl"),
    ("nextgqa-cleaneval", "runs/n5_s100_nextgqa_cleaneval_b4", "data/processed/cleaneval/nextgqa_cleaneval.jsonl"),
    ("nextgqa-cleaneval", "runs/n5_s100_nextgqa_cleaneval_b8", "data/processed/cleaneval/nextgqa_cleaneval.jsonl"),
    ("nextgqa-ho-noscan", "runs/abl_noscan_nextgqa", "data/processed/heldout/nextgqa_heldout.jsonl"),
    ("egoschema-ho-noscan", "runs/abl_noscan_egoschema", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("egoschema-full", "runs/full_v3_egoschema_b16", "data/processed/egoschema_val.jsonl"),
    ("nextgqa-full", "runs/full_v3_nextgqa_b16", "data/processed/nextgqa_val_half.jsonl"),
    ("nextqa-full", "runs/full_v3_nextqa_b16", "data/processed/nextqa_temporal_test_3500_half.jsonl"),
    ("nextgqa-ho", "runs/p4eval_nextgqa_heldout", "data/processed/heldout/nextgqa_heldout.jsonl"),
    ("egoschema-ho", "runs/p4eval_egoschema_heldout", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("lvb-ho", "runs/p4eval_longvideobench_heldout", "data/processed/heldout/longvideobench_heldout.jsonl"),
    ("videomme-ho", "runs/p4eval_videomme_heldout", "data/processed/heldout/videomme_heldout.jsonl"),
    ("nextgqa-full", "runs/full_v3_nextgqa_half_scripted", "data/processed/nextgqa_val_half.jsonl"),
    ("nextqa-full", "runs/full_v3_nextqa_half_scripted", "data/processed/nextqa_temporal_test_3500_half.jsonl"),
    # 2026-07-14 wave: P4 high-n row, full-n selectors, mlvu -scan, pass@32 probe
    ("nextgqa-cleaneval", "runs/p4_nextgqa_cleaneval", "data/processed/cleaneval/nextgqa_cleaneval.jsonl"),
    ("nextgqa-full", "runs/v3sel_full_nextgqa", "data/processed/nextgqa_val_half.jsonl"),
    ("mlvu-full-noscan", "runs/abl_noscan_mlvu", "data/processed/mlvu_val.jsonl"),
    ("nextgqa-ho", "runs/passk32_p4_nextgqa_b8", "data/processed/heldout/nextgqa_heldout_p120.jsonl"),
    # 2026-07-15 wave: question-conditioned WRITE (qcond) vs generic-caption DdE refs
    ("egoschema-ho-qcond", "runs/qcond_egoschema", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("mlvu-ho-qcond", "runs/qcond_mlvu", "data/processed/heldout/mlvu_heldout.jsonl"),
    ("nextgqa-ho-qcond", "runs/qcond_nextgqa", "data/processed/heldout/nextgqa_heldout_p120.jsonl"),
    # 2026-07-18 wave: InternVL cross-family (T1+T3), K=32 ladder, full-n qcond
    ("nextgqa-iv", "runs/xfam_iv_nextgqa", "data/processed/heldout/nextgqa_heldout.jsonl"),
    ("nextqa-iv", "runs/xfam_iv_nextqa", "data/processed/heldout/nextqa_heldout.jsonl"),
    ("egoschema-iv", "runs/xfam_iv_egoschema", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("mlvu-iv", "runs/xfam_iv_mlvu", "data/processed/heldout/mlvu_heldout.jsonl"),
    ("nextgqa-iv", "runs/blind_iv_nextgqa", "data/processed/heldout/nextgqa_heldout.jsonl"),
    ("nextqa-iv", "runs/blind_iv_nextqa", "data/processed/heldout/nextqa_heldout.jsonl"),
    ("egoschema-iv", "runs/blind_iv_egoschema", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("mlvu-iv", "runs/blind_iv_mlvu", "data/processed/heldout/mlvu_heldout.jsonl"),
    ("egoschema-full", "runs/full_v3_egoschema_b32", "data/processed/egoschema_val.jsonl"),
    ("nextgqa-full", "runs/full_v3_nextgqa_b32", "data/processed/nextgqa_val_half.jsonl"),
    ("nextqa-full", "runs/full_v3_nextqa_b32", "data/processed/nextqa_temporal_test_3500_half.jsonl"),
    # qcond full-n runs pair against full_v3_{mlvu,egoschema} dde refs via scripts/qcond_pair.py,
    # not against uniform here (policy name collides with generic dde_curator)
    # 2026-07-19 wave: InternVL mlvu full-n + b16 ladder, Qwen mlvu b32, OneVision third family
    ("mlvu-iv-full", "runs/xfam_iv_mlvu_full", "data/processed/mlvu_val.jsonl"),
    ("mlvu-iv-full-noscan", "runs/abl_noscan_mlvu_iv", "data/processed/mlvu_val.jsonl"),
    ("egoschema-iv-full", "runs/xfam_iv_egoschema_b16", "data/processed/egoschema_val.jsonl"),
    ("egoschema-iv-full", "runs/xfam_iv_egoschema_b32", "data/processed/egoschema_val.jsonl"),
    ("mlvu-full", "runs/full_v3_mlvu_b32", "data/processed/mlvu_val.jsonl"),
    ("nextgqa-ov", "runs/xfam_ov_nextgqa", "data/processed/heldout/nextgqa_heldout.jsonl"),
    ("nextqa-ov", "runs/xfam_ov_nextqa", "data/processed/heldout/nextqa_heldout.jsonl"),
    ("egoschema-ov", "runs/xfam_ov_egoschema", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("mlvu-ov", "runs/xfam_ov_mlvu", "data/processed/heldout/mlvu_heldout.jsonl"),
    ("nextgqa-ov", "runs/blind_ov_nextgqa", "data/processed/heldout/nextgqa_heldout.jsonl"),
    ("nextqa-ov", "runs/blind_ov_nextqa", "data/processed/heldout/nextqa_heldout.jsonl"),
    ("egoschema-ov", "runs/blind_ov_egoschema", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("mlvu-ov", "runs/blind_ov_mlvu", "data/processed/heldout/mlvu_heldout.jsonl"),
    # 2026-07-19b wave
    ("mlvu-full", "runs/full_v3_mlvu_b64", "data/processed/mlvu_val.jsonl"),
    ("egoschema-iv-full", "runs/xfam_iv_egoschema_full", "data/processed/egoschema_val.jsonl"),
    ("egoschema-iv-noscan", "runs/abl_noscan_egoschema_iv", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("mlvu-ov-full", "runs/xfam_ov_mlvu_full", "data/processed/mlvu_val.jsonl"),
    ("nextgqa-ov", "runs/xfam_ov_nextgqa_b16b32", "data/processed/heldout/nextgqa_heldout.jsonl"),
    ("nextqa-ov", "runs/xfam_ov_nextqa_b16b32", "data/processed/heldout/nextqa_heldout.jsonl"),
    # 2026-07-20 IV ladder wave (183867)
    ("nextgqa-iv", "runs/xfam_iv_nextgqa_b16", "data/processed/heldout/nextgqa_heldout.jsonl"),
    ("nextgqa-iv", "runs/xfam_iv_nextgqa_b32", "data/processed/heldout/nextgqa_heldout.jsonl"),
    ("nextqa-iv", "runs/xfam_iv_nextqa_b16", "data/processed/heldout/nextqa_heldout.jsonl"),
    ("nextqa-iv", "runs/xfam_iv_nextqa_b32", "data/processed/heldout/nextqa_heldout.jsonl"),
    ("egoschema-ov", "runs/xfam_ov_egoschema_b16b32", "data/processed/heldout/egoschema_heldout.jsonl"),
    # 2026-07-20 landed wave: Gemma-3 (family 4), uncapped probes (NEVER mix uncap
    # labels with capped panels), seeds, OV mlvu ladder, L2/L5 invariance diagnostics
    ("nextgqa-g3", "runs/xfam_g3_nextgqa", "data/processed/heldout/nextgqa_heldout.jsonl"),
    ("nextqa-g3", "runs/xfam_g3_nextqa", "data/processed/heldout/nextqa_heldout.jsonl"),
    ("egoschema-g3", "runs/xfam_g3_egoschema", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("nextgqa-g3", "runs/blind_g3_nextgqa", "data/processed/heldout/nextgqa_heldout.jsonl"),
    ("nextqa-g3", "runs/blind_g3_nextqa", "data/processed/heldout/nextqa_heldout.jsonl"),
    ("egoschema-g3", "runs/blind_g3_egoschema", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("mlvu-g3", "runs/blind_g3_mlvu", "data/processed/heldout/mlvu_heldout.jsonl"),
    ("mlvu-g3-full", "runs/xfam_g3_mlvu_full", "data/processed/mlvu_val.jsonl"),
    ("mlvu-uncap", "runs/protocol_tax_mlvu", "data/processed/mlvu_val.jsonl"),
    ("nextgqa-uncap", "runs/uncap_nextgqa_full", "data/processed/nextgqa_val_half.jsonl"),
    ("nextgqa-seed0", "runs/seed0_nextgqa_full", "data/processed/nextgqa_val_half.jsonl"),
    ("nextgqa-seed1", "runs/seed1_nextgqa_full", "data/processed/nextgqa_val_half.jsonl"),
    ("mlvu-ov-full", "runs/xfam_ov_mlvu_b16b32", "data/processed/mlvu_val.jsonl"),
    ("l2-led1500-mlvu", "runs/l2_ledger1500_mlvu", "data/processed/heldout/mlvu_heldout.jsonl"),
    ("l2-led3000-mlvu", "runs/l2_ledger3000_mlvu", "data/processed/heldout/mlvu_heldout.jsonl"),
    ("l2-led12000-mlvu", "runs/l2_ledger12000_mlvu", "data/processed/heldout/mlvu_heldout.jsonl"),
    ("l2-led1500-ego", "runs/l2_ledger1500_egoschema", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("l2-led3000-ego", "runs/l2_ledger3000_egoschema", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("l2-led12000-ego", "runs/l2_ledger12000_egoschema", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("l5-hops1-mlvu", "runs/l5_hops1_mlvu", "data/processed/heldout/mlvu_heldout.jsonl"),
    ("l5-hops2-mlvu", "runs/l5_hops2_mlvu", "data/processed/heldout/mlvu_heldout.jsonl"),
    ("l5-hops8-mlvu", "runs/l5_hops8_mlvu", "data/processed/heldout/mlvu_heldout.jsonl"),
    ("l5-hops1-ego", "runs/l5_hops1_egoschema", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("l5-hops2-ego", "runs/l5_hops2_egoschema", "data/processed/heldout/egoschema_heldout.jsonl"),
    ("l5-hops8-ego", "runs/l5_hops8_egoschema", "data/processed/heldout/egoschema_heldout.jsonl"),
    # 2026-07-20b: uncapped chain completion + Gemma ladders (aksoff runs are v2-gen —
    # NEVER register them here)
    ("nextqa-uncap", "runs/uncap_nextqa_full", "data/processed/nextqa_temporal_test_3500_half.jsonl"),
    ("egoschema-uncap", "runs/uncap_egoschema_full", "data/processed/egoschema_val.jsonl"),
    ("nextgqa-g3", "runs/xfam_g3_nextgqa_b16b32", "data/processed/heldout/nextgqa_heldout.jsonl"),
    ("egoschema-g3", "runs/xfam_g3_egoschema_b16b32", "data/processed/heldout/egoschema_heldout.jsonl"),
    # IV mlvu b16 salvage: b32 dropped (no-flash-attn OOM ×2, like ego b32); b16 shards
    # complete in _shards even though the failed job never wrote summary.json
    ("mlvu-iv-full", "runs/xfam_iv_mlvu_b16b32", "data/processed/mlvu_val.jsonl"),
]

SHARD_RE = re.compile(r"b(\d+)_(.+)\.jsonl$")


def wilson(p: float, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - rad) / den, (c + rad) / den)


def mcnemar(a: dict, b: dict):
    common = [q for q in a if q in b]
    x = sum(1 for q in common if a[q] and not b[q])
    y = sum(1 for q in common if b[q] and not a[q])
    if x + y == 0:
        return x, y, 1.0
    p = sum(math.comb(x + y, k) for k in range(0, min(x, y) + 1)) * 2 / 2 ** (x + y)
    return x, y, min(1.0, p)


def load_gold(path: str):
    gold = {}
    for line in open(path):
        q = json.loads(line)
        gold[q["qid"]] = int(q["answer_index"])
    return gold


def load_shard(path: str, gold: dict):
    out = {}
    frames = []
    for line in open(path):
        r = json.loads(line)
        qid = r["qid"]
        if qid not in gold:
            continue
        ai = r.get("answer_index")
        out[qid] = ai is not None and int(ai) == gold[qid]
        frames.append(r.get("num_frames") or 0)
    return out, (sum(frames) / len(frames) if frames else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/sigpack.md")
    args = ap.parse_args()

    # dataset -> budget -> policy -> (accmap, avg_frames, run_dir)
    cells = defaultdict(lambda: defaultdict(dict))
    for label, run_dir, qa in REGISTRY:
        if not os.path.isdir(os.path.join(run_dir, "_shards")):
            continue
        gold = load_gold(qa)
        for sh in sorted(glob.glob(os.path.join(run_dir, "_shards", "*.jsonl"))):
            m = SHARD_RE.search(os.path.basename(sh))
            if not m:
                continue
            budget, policy = int(m.group(1)), m.group(2)
            accmap, avg_f = load_shard(sh, gold)
            if accmap:
                cells[label][budget][policy] = (accmap, avg_f, run_dir)

    lines = ["# Significance pack (auto-generated by scripts/sigpack.py)", ""]
    for label in sorted(cells):
        lines.append(f"## {label}")
        for budget in sorted(cells[label]):
            pols = cells[label][budget]
            lines.append(f"\n### budget {budget}")
            lines.append("| policy | acc | 95% CI | frames | n | vs uniform Δ | disc | McNemar p |")
            lines.append("|---|---|---|---|---|---|---|---|")
            uni = pols.get("uniform")
            for pol in sorted(pols):
                accmap, avg_f, _ = pols[pol]
                n = len(accmap)
                acc = sum(accmap.values()) / n
                lo, hi = wilson(acc, n)
                if uni is not None and pol != "uniform":
                    ua = uni[0]
                    common = [q for q in accmap if q in ua]
                    if common:
                        d = (sum(accmap[q] for q in common) - sum(ua[q] for q in common)) / len(common)
                        x, y, p = mcnemar(accmap, ua)
                        extra = f"| {d:+.3f} | {x}/{y} | {p:.3f}{' *' if p < 0.05 else ''} |"
                    else:
                        extra = "| — | — | — |"
                else:
                    extra = "| — | — | — |"
                lines.append(
                    f"| {pol} | {acc:.3f} | [{lo:.3f},{hi:.3f}] | {avg_f:.1f} | {n} {extra}"
                )
        lines.append("")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({sum(len(b) for b in cells.values())} dataset-budget panels)")


if __name__ == "__main__":
    main()
