"""N3 — deployable question-text locality router (zero GPU).

W5 fix: the C3 router routes by benchmark-provided question-type labels ("oracle-ish" at
deploy time). Here we predict the A-PRIORI locality class from the question TEXT alone
(SigLIP text embedding, the same one retrieval already computes), so routing needs no
benchmark metadata.

This is NOT the failed Result-1 probe (router.md): that probe predicted per-question NET
BENEFIT (a noisy label mixing locality with answerer/retrieval noise) and landed at
AUROC~=0.5. The target here is the question TYPE's preregistered locality label — a
deterministic property of how the question is phrased.

Part A (gate): numpy logistic head on query_embedding -> P(localizable),
leave-one-dataset-out; pooled AUROC over held-out predictions. GATE-N3: AUROC >= 0.75.
Part B: downstream routing on the strict firewall held-out — route each question
(P>=0.5 -> scripted_curator, else learned_curator) using EXISTING per-question
predictions; report routed CE vs single policies vs the type-oracle router.

A-priori labels come from the SAME taxonomy X1 preregistered (router.md Result 2 /
x1_prospective_validation.md); ambiguous/mixed types are EXCLUDED from training and
AUROC (they still get routed in Part B).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# dataset -> (siglip qa file with query_embedding, how to read the type, type->label map)
# label 1 = localizable, 0 = global; type absent from the map = excluded (ambiguous a-priori)
SPECS = {
    "nextgqa": dict(
        qa="data/processed/nextgqa_val_siglip.jsonl",
        type_key=("metadata", "type"),
        labels=defaultdict(lambda: 1),  # whole dataset preregistered localizable
    ),
    "nextqa": dict(
        qa=["data/processed/nextqa_temporal_test_3500_siglip.jsonl",
            "data/processed/nextqa_temporal_val_450_siglip.jsonl"],
        type_key=("metadata", "type"),
        labels=defaultdict(lambda: 1),
    ),
    "egoschema": dict(
        qa="data/processed/egoschema_val_siglip.jsonl",
        type_key=None,
        labels=defaultdict(lambda: 0),  # whole dataset preregistered global
    ),
    "mlvu": dict(
        qa="data/processed/mlvu_val_siglip.jsonl",
        type_key=("metadata", "raw", "question_type"),
        labels={"needleQA": 1, "ego": 1, "sportsQA": 0, "anomaly_reco": 0,
                "topic_reasoning": 0, "plotQA": 0, "order": 0},  # count excluded (mixed)
    ),
    "videomme": dict(
        qa="data/processed/videomme_long_siglip.jsonl",
        type_key=("metadata", "task_type"),
        labels={"Object Recognition": 1, "Action Recognition": 1, "OCR Problems": 1,
                "Attribute Perception": 1, "Information Synopsis": 0,
                "Temporal Reasoning": 0},  # Action/Object Reasoning, Counting excluded
    ),
    "longvideobench": dict(
        qa="data/processed/longvideobench_val_siglip.jsonl",
        type_key=("metadata", "raw", "question_category"),
        labels={"T2A": 1, "S2E": 1, "T2E": 1, "T2O": 1, "S2O": 1, "E2O": 1, "O2E": 1,
                "S2A": 0, "SSS": 0, "SAA": 0, "TOS": 0, "TAA": 0},
    ),
    "lvbench": dict(
        qa="data/processed/lvbench_val_siglip.jsonl",
        type_key=("metadata", "raw", "category"),
        labels={"key information retrieval": 1, "temporal grounding": 1,
                "entity recognition": 1, "summarization": 0},
        # event understanding / reasoning excluded (preregistered "0"/neutral)
    ),
}

# Part B: held-out prediction sources (scripted from r2_v2 full runs, learned from
# s1181 camera-ready runs), budgets intersected per dataset.
PARTB = {
    "videomme": dict(scripted="runs/r2_v2_videomme_qwen2_5_full",
                     learned="runs/s1181_videomme_453", budgets=[4, 8, 16, 32]),
    "longvideobench": dict(scripted="runs/r2_v2_longvideobench_qwen2_5_full",
                           learned="runs/s1181_lvb_669", budgets=[4, 8, 16, 32]),
    "egoschema": dict(scripted="runs/r2_v2_egoschema_qwen2_5_full",
                      learned="runs/s1181_egoschema_252", budgets=[4, 8, 16, 32]),
    "mlvu": dict(scripted="runs/r2_v2_mlvu_qwen2_5_full",
                 learned="runs/s1181_mlvu_253", budgets=[4, 8, 16, 32]),
    "nextgqa": dict(scripted="runs/gpu_nextgqa_qwen2_5_learned_grpo_pow452",
                    learned="runs/s1181_nextgqa_452", budgets=[4, 8, 16]),
    "nextqa": dict(scripted="runs/gpu_nextqa_qwen2_5_learned_grpo_pow452",
                   learned="runs/s1181_nextqa_452", budgets=[4, 8, 16]),
}


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_nested(d: dict, keys) -> object:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def qa_files(spec: dict):
    files = spec["qa"] if isinstance(spec["qa"], list) else [spec["qa"]]
    return [ROOT / f for f in files]


def iter_spec_rows(spec: dict):
    seen = set()
    for path in qa_files(spec):
        for row in iter_jsonl(path):
            q = str(row.get("qid"))
            if q in seen:
                continue
            seen.add(q)
            yield row


def load_dataset(ds: str, spec: dict):
    """Return (qids, X, y) for labeled rows; y=None rows are skipped."""
    qids, X, y = [], [], []
    for row in iter_spec_rows(spec):
        md = row.get("metadata") or {}
        emb = md.get("query_embedding")
        if emb is None:
            continue
        if spec["type_key"] is None:
            label = spec["labels"]["_"]
        else:
            t = get_nested(row, spec["type_key"])
            labels = spec["labels"]
            if isinstance(labels, defaultdict):
                label = labels[t]
            elif t in labels:
                label = labels[t]
            else:
                continue
        qids.append(str(row["qid"]))
        X.append(np.asarray(emb, dtype=np.float32))
        y.append(label)
    return qids, np.stack(X) if X else np.zeros((0, 1)), np.asarray(y, dtype=np.float32)


def load_embeddings(ds: str, spec: dict):
    """qid -> embedding for ALL rows (Part B routes unlabeled questions too)."""
    out = {}
    for row in iter_spec_rows(spec):
        emb = (row.get("metadata") or {}).get("query_embedding")
        if emb is not None:
            out[str(row["qid"])] = np.asarray(emb, dtype=np.float32)
    return out


def train_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2,
                   iters: int = 300, lr: float = 0.5):
    """Class-balanced L2 logistic regression, full-batch gradient descent."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    n, d = Xn.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    # balance: weight each class inversely to its frequency
    w_pos = n / (2.0 * max(y.sum(), 1.0))
    w_neg = n / (2.0 * max((1 - y).sum(), 1.0))
    sw = np.where(y > 0.5, w_pos, w_neg)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xn @ w + b)))
        g = (p - y) * sw
        gw = Xn.T @ g / n + l2 * w
        gb = g.mean()
        w -= lr * gw
        b -= lr * gb
    return w, b, mu, sd


def predict(w, b, mu, sd, X):
    Xn = (X - mu) / sd
    return 1.0 / (1.0 + np.exp(-(Xn @ w + b)))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n_pos, n_neg = int((y > 0.5).sum()), int((y <= 0.5).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y > 0.5].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def load_predictions(run_dir: str, policy: str, budgets, gold: dict):
    """(qid, budget) -> correct for one policy, consolidated-if-present rule."""
    run = ROOT / run_dir
    path = run / "predictions.jsonl"
    rows = list(iter_jsonl(path)) if path.exists() else []
    if not rows:
        for shard in sorted((run / "_shards").glob("*.jsonl")):
            rows.extend(iter_jsonl(shard))
    out = {}
    for r in rows:
        if r.get("policy") != policy:
            continue
        bud = int(r.get("budget", -1))
        if bud not in budgets:
            continue
        q = str(r["qid"])
        if q not in gold:
            continue
        pred = r.get("answer_index")
        out[(q, bud)] = float(pred is not None and int(pred) == gold[q])
    return out


def load_gold(qa_file: str) -> dict:
    gold = {}
    for row in iter_jsonl(ROOT / qa_file):
        if row.get("answer_index") is not None:
            gold[str(row["qid"])] = int(row["answer_index"])
    return gold


def main() -> None:
    print("=== Part A: LODO logistic on a-priori locality labels ===")
    data = {ds: load_dataset(ds, spec) for ds, spec in SPECS.items()}
    for ds, (q, X, y) in data.items():
        print(f"  {ds}: n_labeled={len(q)} pos_frac={y.mean() if len(y) else float('nan'):.2f} dim={X.shape[1] if len(q) else 0}")

    emb_all = {ds: load_embeddings(ds, spec) for ds, spec in SPECS.items()}
    pooled_y, pooled_s, per_ds = [], [], {}
    mixed_y, mixed_s = [], []  # within-mixed-datasets pool (both classes inside each ds)
    lodo_scores = {}  # ds -> {qid: score} from the model that never saw ds (ALL questions)
    for held in SPECS:
        train_X = np.concatenate([data[d][1] for d in SPECS if d != held and len(data[d][0])])
        train_y = np.concatenate([data[d][2] for d in SPECS if d != held and len(data[d][0])])
        w, b, mu, sd = train_logistic(train_X, train_y)
        all_q = list(emb_all[held].keys())
        if all_q:
            s_all = predict(w, b, mu, sd, np.stack([emb_all[held][q] for q in all_q]))
            lodo_scores[held] = dict(zip(all_q, s_all))
        qids, X, y = data[held]
        if not len(qids):
            continue
        s = predict(w, b, mu, sd, X)
        pooled_y.append(y)
        pooled_s.append(s)
        if 0.0 < y.mean() < 1.0:  # mixed dataset: within-ds AUROC is meaningful
            mixed_y.append(y)
            mixed_s.append(s)
        per_ds[held] = dict(n=len(y), auroc=auroc(y, s), mean_score=float(s.mean()),
                            frac_routed_loc=float((s >= 0.5).mean()))
    ya, sa = np.concatenate(pooled_y), np.concatenate(pooled_s)
    pooled = auroc(ya, sa)
    acc05 = float((((sa >= 0.5) == (ya > 0.5))).mean())
    ym, sm = np.concatenate(mixed_y), np.concatenate(mixed_s)
    mixed_auroc = auroc(ym, sm)

    lines = ["# N3 — question-text locality router (deployable C3)", "",
             "_Logistic head on SigLIP question-text embeddings; a-priori type labels",
             "(same taxonomy X1 preregistered); leave-one-dataset-out. Ambiguous types",
             "excluded from labels. NOT the failed net-benefit probe (router.md R1)._", "",
             f"**Pooled LODO AUROC = {pooled:.3f}** (gate >= 0.75), accuracy@0.5 = {acc05:.3f}, n = {len(ya)}",
             f"**Within-mixed-datasets LODO AUROC = {mixed_auroc:.3f}** (n = {len(ym)}; the stringent",
             "number — pooled AUROC partly reflects dataset identity; this one cannot).", "",
             "| held-out dataset | n | AUROC | mean P(loc) | frac routed->scripted |", "|---|---|---|---|---|"]
    for ds, r in per_ds.items():
        au = "n/a (single-class)" if np.isnan(r["auroc"]) else f"{r['auroc']:.3f}"
        lines.append(f"| {ds} | {r['n']} | {au} | {r['mean_score']:.3f} | {r['frac_routed_loc']:.2f} |")
    print("\n".join(lines[5:]))

    # ---- Part B: downstream routing on strict firewall held-out --------------------
    print("\n=== Part B: routed CE on firewall held-out ===")
    fw = json.loads((ROOT / "data/rl/firewall_split.json").read_text())
    heldout = {ds: set(spec["heldout_qids"]) for ds, spec in fw["eval"].items()}
    gold_maps = {ds: load_gold(spec["file"]) for ds, spec in fw["eval"].items()}

    lines += ["", "## Downstream: route P>=0.5 -> scripted, else learned (strict held-out)",
              "", "| dataset | n paired | uniform | scripted | learned | ROUTED | type-oracle(ds-level) |",
              "|---|---|---|---|---|---|---|"]
    A_PRIORI_DS = {"nextgqa": 1, "nextqa": 1, "mlvu": 1, "videomme": 0,
                   "longvideobench": 0, "egoschema": 0}
    for ds, pb in PARTB.items():
        gold = gold_maps[ds]
        scripted = load_predictions(pb["scripted"], "scripted_curator", pb["budgets"], gold)
        learned = load_predictions(pb["learned"], "learned_curator", pb["budgets"], gold)
        uniform = load_predictions(pb["scripted"], "uniform", pb["budgets"], gold)
        scores = lodo_scores.get(ds, {})
        common_q = ({q for q, _ in scripted} & {q for q, _ in learned}
                    & {q for q, _ in uniform} & heldout.get(ds, set()))
        rows = []
        missing_emb = 0
        for q in common_q:
            s = scores.get(q)
            if s is None:
                missing_emb += 1
                continue
            rows.append((q, s))
        per_policy = {k: [] for k in ("uniform", "scripted", "learned", "routed", "oracle")}
        for q, s in rows:
            cells_s = [scripted.get((q, b)) for b in pb["budgets"]]
            cells_l = [learned.get((q, b)) for b in pb["budgets"]]
            cells_u = [uniform.get((q, b)) for b in pb["budgets"]]
            if None in cells_s or None in cells_l or None in cells_u:
                continue
            ce_s, ce_l, ce_u = np.mean(cells_s), np.mean(cells_l), np.mean(cells_u)
            per_policy["uniform"].append(ce_u)
            per_policy["scripted"].append(ce_s)
            per_policy["learned"].append(ce_l)
            per_policy["routed"].append(ce_s if (s is not None and s >= 0.5) else ce_l)
            per_policy["oracle"].append(ce_s if A_PRIORI_DS[ds] else ce_l)
        n = len(per_policy["uniform"])
        if not n:
            print(f"  {ds}: no paired rows (missing_emb={missing_emb}) — skipped")
            continue
        m = {k: float(np.mean(v)) for k, v in per_policy.items()}
        lines.append(f"| {ds} | {n} | {m['uniform']:.3f} | {m['scripted']:.3f} | "
                     f"{m['learned']:.3f} | **{m['routed']:.3f}** | {m['oracle']:.3f} |")
        print(f"  {ds}: n={n} routed={m['routed']:.3f} oracle={m['oracle']:.3f} "
              f"scripted={m['scripted']:.3f} learned={m['learned']:.3f} uniform={m['uniform']:.3f}")

    verdict = "PASS" if pooled >= 0.75 else "FAIL"
    lines += ["", f"**GATE-N3 (pooled LODO AUROC >= 0.75): {verdict} ({pooled:.3f})**"]
    out = ROOT / "reports/n3_question_router.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWROTE {out}  |  GATE-N3: {verdict} ({pooled:.3f})")


if __name__ == "__main__":
    main()
