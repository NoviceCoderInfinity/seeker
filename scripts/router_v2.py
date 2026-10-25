#!/usr/bin/env python3
"""Router v2: question-text portfolio router over {blind, uniform@8, dde@8} (zero GPU).

Headroom-gated learning: RL on curation/answer turns is measured dead (dossier §16),
but the 3-arm per-question oracle beats the best single arm by +0.12-0.18 on every
dataset. This script measures how much of that ceiling a question-text router realizes.

Firewall: trains ONLY on nextgqa_traincln + egoschema_traincln qids; evaluates on
(a) nextgqa cleaneval-2707 (within-dataset, video+question disjoint),
(b) mlvu full and nextqa half as ENTIRELY UNSEEN datasets (cross-dataset transfer).
Any eval qid colliding with a train qid is dropped defensively.

Model: TF-IDF (word 1-2gram + char_wb 3-5gram) -> one logistic regression per arm
predicting P(correct); router picks argmax, ties break to dde (the deployable default).
Reports accuracy vs best single arm + McNemar, and realized fraction of the 3-arm oracle.
"""
from __future__ import annotations

import json
from math import comb

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ARMS = ["blind", "uniform8", "dde8"]  # index = router class; ties -> dde8 (last wins)

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
        opts = " | ".join(r.get("options") or [])
        out[r["qid"]] = (r["answer_index"], f"{r['question']} || {opts}")
    return out


def load_arm(path, gold):
    d = {}
    for line in open(path):
        r = json.loads(line)
        if r["qid"] in gold:
            d[r["qid"]] = r.get("answer_index") is not None and int(r["answer_index"]) == gold[r["qid"]][0]
    return d


def build_rows(name):
    spec = SHARDS[name]
    gold = load_gold(spec["gold"])
    arms = {a: load_arm(spec[a], gold) for a in ARMS}
    qids = sorted(set.intersection(*[set(arms[a]) for a in ARMS]))
    return [(q, gold[q][1], [arms[a][q] for a in ARMS]) for q in qids]


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
    print(f"train rows: {len(train_rows)} (nextgqa+egoschema traincln ∩ all-3-arm coverage)")

    texts = [t for _, t, _ in train_rows]
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    Xw, Xc = word.fit_transform(texts), char.fit_transform(texts)
    X = hstack([Xw, Xc]).tocsr()

    clfs = []
    for i, arm in enumerate(ARMS):
        y = np.array([oks[i] for _, _, oks in train_rows], dtype=int)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(X, y)
        clfs.append(clf)
        print(f"  {arm}: train base rate {y.mean():.3f}")

    lines = ["# Router v2 — question-text portfolio router (train: nextgqa+ego traincln)",
             "", "| eval set | n | " + " | ".join(ARMS) +
             " | best arm | router | oracle3 | realized | router vs best p |", "|" + "---|" * 9]
    for name in ("nextgqa_cleaneval", "mlvu", "nextqa", "egoschema"):
        rows = [r for r in build_rows(name) if r[0] not in train_qids]
        if not rows:
            print(f"{name}: no eval rows"); continue
        Xe = hstack([word.transform([t for _, t, _ in rows]),
                     char.transform([t for _, t, _ in rows])]).tocsr()
        proba = np.column_stack([c.predict_proba(Xe)[:, 1] for c in clfs])
        # argmax with tie-break to LAST index (dde8): reverse argmax on flipped array
        pick = proba.shape[1] - 1 - np.argmax(proba[:, ::-1] + 1e-12 * 0, axis=1)
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

    with open("reports/router_v2.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote reports/router_v2.md")


if __name__ == "__main__":
    main()
