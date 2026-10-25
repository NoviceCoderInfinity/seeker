#!/usr/bin/env python3
"""Router v3: MEMORY-STATE-conditioned arbitration over {blind, uniform@8, dde@8}.

Router v2 (question text only) realized NEGATIVE fractions of the 3-arm oracle
(+0.12-0.18 available). v3 tests the memory-design hypothesis: the curator's own
trace — probe sufficiency, hop count, retrieval-score dispersal, ledger shape —
plus the three arms' answer-agreement pattern is a *state* signal the question
text cannot carry. Deploy cost is 3 arms (already priced); the arbiter is free.

Firewall identical to v2: train ONLY on nextgqa+egoschema traincln qids; evaluate
on nextgqa cleaneval and on mlvu/nextqa/egoschema-minus-train as unseen sets.
Kill criterion (pre-registered, one iteration): beat the best single arm on
cleaneval with McNemar p<0.05 AND transfer positively to >=1 unseen dataset;
otherwise the arbitration lane is closed for the paper like every other lever.
"""
from __future__ import annotations

import json
from math import comb

import numpy as np
from sklearn.linear_model import LogisticRegression

ARMS = ["blind", "uniform8", "dde8"]

SHARDS = {
    "nextgqa": {
        "blind": "runs/blind_full_nextgqa/_shards/b0_uniform.jsonl",
        "uniform8": "runs/full_v3_nextgqa_half/_shards/b8_uniform.jsonl",
        "dde8": "runs/full_v3_nextgqa_half/_shards/b8_dde_curator.jsonl",
        "gold": "data/processed/nextgqa_val.jsonl",
    },
    "nextgqa_cleaneval": {
        "blind": "runs/blind_full_nextgqa/_shards/b0_uniform.jsonl",
        "uniform8": "runs/full_v3_nextgqa_cleaneval/_shards/b8_uniform.jsonl",
        "dde8": "runs/full_v3_nextgqa_cleaneval/_shards/b8_dde_curator.jsonl",
        "gold": "data/processed/nextgqa_val.jsonl",
    },
    "egoschema": {
        "blind": "runs/blind_full_egoschema/_shards/b0_uniform.jsonl",
        "uniform8": "runs/full_v3_egoschema/_shards/b8_uniform.jsonl",
        "dde8": "runs/full_v3_egoschema/_shards/b8_dde_curator.jsonl",
        "gold": "data/processed/egoschema_val.jsonl",
    },
    "mlvu": {
        "blind": "runs/blind_full_mlvu/_shards/b0_uniform.jsonl",
        "uniform8": "runs/full_v3_mlvu/_shards/b8_uniform.jsonl",
        "dde8": "runs/full_v3_mlvu/_shards/b8_dde_curator.jsonl",
        "gold": "data/processed/mlvu_val.jsonl",
    },
    "nextqa": {
        "blind": "runs/blind_full_nextqa/_shards/b0_uniform.jsonl",
        "uniform8": "runs/full_v3_nextqa_half/_shards/b8_uniform.jsonl",
        "dde8": "runs/full_v3_nextqa_half/_shards/b8_dde_curator.jsonl",
        "gold": "data/processed/nextqa_temporal_test_3500.jsonl",
    },
}


def load_gold(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["qid"]] = r["answer_index"]
    return out


def load_arm(path, gold, with_state=False):
    """qid -> (answer_index, correct[, state-features])."""
    d = {}
    for line in open(path):
        r = json.loads(line)
        q = r["qid"]
        if q not in gold:
            continue
        ai = r.get("answer_index")
        ok = ai is not None and int(ai) == gold[q]
        if not with_state:
            d[q] = (ai, ok)
            continue
        suff, hops, ret_scores = 0.0, 0, []
        for ev in r.get("trace", []):
            if ev.get("event") == "probe":
                suff = float((ev.get("probe") or {}).get("sufficiency", suff))
                hops = max(hops, int(ev.get("hop", 0)))
            elif ev.get("event") == "retrieve":
                ret_scores += [float(f.get("score", 0.0)) for f in ev.get("frames", [])]
        led = len((r.get("metadata") or {}).get("running_summary", ""))
        rs = np.array(ret_scores) if ret_scores else np.zeros(1)
        state = [suff, hops, r.get("num_retrievals", 0) or 0, r.get("num_frames", 0) or 0,
                 led / 1000.0, float(rs.max()), float(rs.mean()),
                 float((rs > 0).sum()), float(rs.std())]
        d[q] = (ai, ok, state)
    return d


def build_rows(name):
    spec = SHARDS[name]
    gold = load_gold(spec["gold"])
    blind = load_arm(spec["blind"], gold)
    uni = load_arm(spec["uniform8"], gold)
    dde = load_arm(spec["dde8"], gold, with_state=True)
    qids = sorted(set(blind) & set(uni) & set(dde))
    rows = []
    for q in qids:
        b_ai, b_ok = blind[q]
        u_ai, u_ok = uni[q]
        d_ai, d_ok, state = dde[q]
        def eq(x, y):
            return float(x is not None and y is not None and int(x) == int(y))
        feats = state + [eq(b_ai, u_ai), eq(b_ai, d_ai), eq(u_ai, d_ai),
                         eq(b_ai, u_ai) * eq(u_ai, d_ai)]
        rows.append((q, feats, [b_ok, u_ok, d_ok]))
    return rows


def mcnemar(a_ok, b_ok):
    b10 = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    b01 = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    m = b10 + b01
    if m == 0:
        return b10, b01, 1.0
    p = min(1.0, 2 * sum(comb(m, i) for i in range(min(b10, b01) + 1)) / 2 ** m)
    return b10, b01, p


def main():
    train_qids = set()
    for p in ("data/processed/traincln/nextgqa_traincln.jsonl",
              "data/processed/traincln/egoschema_traincln.jsonl"):
        for line in open(p):
            train_qids.add(json.loads(line)["qid"])

    train_rows = [r for name in ("nextgqa", "egoschema")
                  for r in build_rows(name) if r[0] in train_qids]
    print(f"train rows: {len(train_rows)}")
    X = np.array([f for _, f, _ in train_rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mu) / sd

    clfs = []
    for i, arm in enumerate(ARMS):
        y = np.array([oks[i] for _, _, oks in train_rows], dtype=int)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(Xn, y)
        clfs.append(clf)
        print(f"  {arm}: train base rate {y.mean():.3f}")

    lines = ["# Router v3 — memory-state arbitration (features: dde trace + arm agreement)",
             "", "| eval set | n | " + " | ".join(ARMS) +
             " | best arm | router | oracle3 | realized | router vs best p |", "|" + "---|" * 9]
    for name in ("nextgqa_cleaneval", "mlvu", "nextqa", "egoschema"):
        rows = [r for r in build_rows(name) if r[0] not in train_qids]
        if not rows:
            continue
        Xe = (np.array([f for _, f, _ in rows]) - mu) / sd
        proba = np.column_stack([c.predict_proba(Xe)[:, 1] for c in clfs])
        pick = np.where(np.abs(proba.max(axis=1, keepdims=True) - proba).sum(axis=1) == 0,
                        len(ARMS) - 1, np.argmax(proba, axis=1))
        oks = np.array([r[2] for r in rows], dtype=bool)
        arm_acc = oks.mean(axis=0)
        best_i = int(np.argmax(arm_acc))
        router_ok = oks[np.arange(len(rows)), pick]
        oracle = oks.any(axis=1).mean()
        realized = (router_ok.mean() - arm_acc[best_i]) / max(oracle - arm_acc[best_i], 1e-9)
        b10, b01, p = mcnemar(router_ok.tolist(), oks[:, best_i].tolist())
        note = "*" if p < 0.05 else ""
        print(f"{name}: n={len(rows)} arms={[f'{a:.3f}' for a in arm_acc]} best={ARMS[best_i]} "
              f"router={router_ok.mean():.3f} oracle={oracle:.3f} realized={realized:+.1%} "
              f"disc={b10}/{b01} p={p:.4f}{note}")
        lines.append(f"| {name} | {len(rows)} | " +
                     " | ".join(f"{a:.3f}" for a in arm_acc) +
                     f" | {ARMS[best_i]} {arm_acc[best_i]:.3f} | {router_ok.mean():.3f} "
                     f"| {oracle:.3f} | {realized:+.1%} | {b10}/{b01} p={p:.4f}{note} |")

    with open("reports/router_v3.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote reports/router_v3.md")


if __name__ == "__main__":
    main()
