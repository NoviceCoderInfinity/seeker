#!/usr/bin/env python3
"""Paired McNemar: full-n question-conditioned WRITE (qcond) vs generic-caption DdE.

qcond runs use policy name dde_curator (scan_conditioning=true in config), so they
cannot be paired inside sigpack (name collides with the generic dde_curator reference).
Pairs runs/qcond_full_<ds> against runs/full_v3_<ds> b{4,8}_dde_curator shards on
identical qids, scored against gold.
"""
import json, math, glob, os

CELLS = [
    ("mlvu", "runs/qcond_full_mlvu", "runs/full_v3_mlvu", "data/processed/mlvu_val.jsonl"),
    ("egoschema", "runs/qcond_full_egoschema", "runs/full_v3_egoschema", "data/processed/egoschema_val.jsonl"),
]

def gold_map(p):
    return {json.loads(l)["qid"]: int(json.loads(l)["answer_index"]) for l in open(p)}

def acc_map(shard, gold):
    out = {}
    for l in open(shard):
        r = json.loads(l)
        q = r["qid"]
        if q in gold:
            ai = r.get("answer_index")
            out[q] = ai is not None and int(ai) == gold[q]
    return out

def mcnemar(a, b):
    common = [q for q in a if q in b]
    x = sum(1 for q in common if a[q] and not b[q])
    y = sum(1 for q in common if b[q] and not a[q])
    if x + y == 0:
        return x, y, 1.0, len(common)
    p = sum(math.comb(x + y, k) for k in range(0, min(x, y) + 1)) * 2 / 2 ** (x + y)
    return x, y, min(1.0, p), len(common)

print("| dataset | budget | qcond | generic DdE | paired Δ | disc (q+/g+) | McNemar p | n |")
print("|---|---|---|---|---|---|---|---|")
for ds, qdir, gdir, goldp in CELLS:
    gold = gold_map(goldp)
    for b in (4, 8):
        qs = os.path.join(qdir, "_shards", f"b{b}_dde_curator.jsonl")
        gs = os.path.join(gdir, "_shards", f"b{b}_dde_curator.jsonl")
        if not (os.path.exists(qs) and os.path.exists(gs)):
            print(f"| {ds} | {b} | MISSING SHARD |")
            continue
        qa, ga = acc_map(qs, gold), acc_map(gs, gold)
        common = [q for q in qa if q in ga]
        qacc = sum(qa[q] for q in common) / len(common)
        gacc = sum(ga[q] for q in common) / len(common)
        x, y, p, n = mcnemar(qa, ga)
        star = " *" if p < 0.05 else ""
        print(f"| {ds} | {b} | {qacc:.4f} | {gacc:.4f} | {qacc-gacc:+.4f} | {x}/{y} | {p:.4f}{star} | {n} |")
