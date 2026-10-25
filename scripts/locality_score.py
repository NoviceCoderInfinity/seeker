"""W-ROUTER (plan4 §3.2): per-question evidence-locality score + validation.

Computes, for every question in the six R2 zero-shot runs, a *locality score* from the
SigLIP query->frame cosine-similarity profile (zero GPU: query embeddings live in the
QA `*_siglip.jsonl` metadata, frame embeddings live in the `*_siglip_frames.jsonl` index).

Three candidate scores (higher = more localizable):
  topk_mass  — sum of top-8 softmax(z) probability mass (z = standardized sims)
  negentropy — 1 - H(softmax(z)) / ln(N)
  peak_frac  — 1 - fraction of frames within the top 10% of the sim range (peak narrowness)

Validation: paired per-qid outcomes from predictions.jsonl (scripted_curator vs uniform at
each budget). On *discordant* pairs (exactly one policy correct), AUROC of score for
"curator right, uniform wrong". Gate G1: pooled AUROC >= 0.62 for the best score.

Outputs: reports/router_scores.csv + stdout summary.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "nextgqa": dict(
        qa="data/processed/nextgqa_val_siglip.jsonl",
        index="data/indices/nextgqa_siglip_frames.jsonl",
        run="runs/gpu_nextgqa_qwen2_5_zero_shot",
        group="localizable",
    ),
    "nextqa": dict(
        qa="data/processed/nextqa_temporal_val_150_siglip.jsonl",
        index="data/indices/nextqa_temporal_siglip_frames.jsonl",
        run="runs/gpu_nextqa_qwen2_5_zero_shot",
        group="localizable",
    ),
    "egoschema": dict(
        qa="data/processed/egoschema_val_siglip.jsonl",
        index="data/indices/egoschema_siglip_frames.jsonl",
        run="runs/gpu_egoschema_qwen2_5_zero_shot",
        group="global",
    ),
    "mlvu": dict(
        qa="data/processed/mlvu_val_siglip.jsonl",
        index="data/indices/mlvu_siglip_frames.jsonl",
        index_alt="data/indices/mlvu_150_siglip_frames.jsonl",
        run="runs/gpu_mlvu_qwen2_5_zero_shot",
        group="global",
    ),
    "videomme": dict(
        qa="data/processed/videomme_long_siglip.jsonl",
        index="data/indices/videomme_long_siglip_frames.jsonl",
        run="runs/gpu_videomme_qwen2_5_zero_shot",
        group="global",
    ),
    "longvideobench": dict(
        qa="data/processed/longvideobench_val_siglip.jsonl",
        index="data/indices/longvideobench_siglip_frames.jsonl",
        run="runs/gpu_longvideobench_qwen2_5_zero_shot",
        group="global",
    ),
}

BUDGETS = [2, 4, 8, 16, 32]
TOPK = 8


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_predictions(run_dir: Path):
    """qid -> {(policy, budget): correct_pred_index}"""
    preds = defaultdict(dict)
    merged = run_dir / "predictions.jsonl"
    files = [merged] if merged.exists() else sorted((run_dir / "_shards").glob("b*_*.jsonl"))
    for path in files:
        for d in iter_jsonl(path):
            preds[str(d["qid"])][(d["policy"], d["budget"])] = d.get("answer_index")
    return preds


def scores_from_sims(sims: np.ndarray) -> dict:
    n = len(sims)
    if n < 4:
        return {}
    std = sims.std()
    z = (sims - sims.mean()) / (std + 1e-8)
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()
    top = np.sort(p)[::-1]
    topk_mass = float(top[: min(TOPK, n)].sum())
    ent = float(-(p * np.log(p + 1e-12)).sum())
    negentropy = float(1.0 - ent / math.log(n))
    rng = sims.max() - sims.min()
    if rng < 1e-8:
        peak_frac = 0.0
    else:
        peak_frac = float(1.0 - (sims >= sims.max() - 0.1 * rng).mean())
    return dict(topk_mass=topk_mass, negentropy=negentropy, peak_frac=peak_frac, n_frames=n)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC via rank statistic (ties averaged)."""
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    rows = []
    for ds, spec in DATASETS.items():
        run_dir = ROOT / spec["run"]
        preds = load_predictions(run_dir)
        qids = set(preds)

        qa_rows = {}
        for d in iter_jsonl(ROOT / spec["qa"]):
            qid = str(d.get("qid"))
            if qid in qids:
                qa_rows[qid] = d
        missing_qa = len(qids) - len(qa_rows)

        # Path().stem on both sides — same rule as vidcurator.retrieval.normalize_video_id
        needed_videos = {Path(str(d["video_id"])).stem for d in qa_rows.values()}
        frames = defaultdict(list)
        for idx_key in ("index", "index_alt"):
            if idx_key not in spec:
                continue
            for d in iter_jsonl(ROOT / spec[idx_key]):
                vid = Path(str(d["video_id"])).stem
                if vid in needed_videos and d.get("embedding"):
                    frames[vid].append(d["embedding"])
            if len(frames) == len(needed_videos):
                break

        n_scored = 0
        for qid, d in qa_rows.items():
            emb = (d.get("metadata") or {}).get("query_embedding")
            vid = Path(str(d["video_id"])).stem
            if not emb or vid not in frames:
                continue
            q = np.asarray(emb, dtype=np.float32)
            F = np.asarray(frames[vid], dtype=np.float32)
            qn = q / (np.linalg.norm(q) + 1e-8)
            Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
            sc = scores_from_sims(Fn @ qn)
            if not sc:
                continue
            row = dict(dataset=ds, group=spec["group"], qid=qid, **sc)
            gold = d.get("answer_index", d.get("gold_index"))
            for b in BUDGETS:
                cur = preds[qid].get(("scripted_curator", b))
                uni = preds[qid].get(("uniform", b))
                row[f"cur_correct@{b}"] = None if cur is None and gold is None else int(cur == gold)
                row[f"uni_correct@{b}"] = None if uni is None and gold is None else int(uni == gold)
            rows.append(row)
            n_scored += 1
        print(f"{ds:16} qids={len(qids)} qa_matched={len(qa_rows)} (missing {missing_qa}) scored={n_scored}")

    out = ROOT / "reports/router_scores.csv"
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("dataset", "group", "qid"), k))
    with out.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + "\n")
    print(f"\nwrote {out} ({len(rows)} rows)")

    # dataset-level score means (does the per-question score recover the known boundary?)
    print("\n=== mean locality score by dataset (expect localizable > global) ===")
    for score in ("topk_mass", "negentropy", "peak_frac"):
        parts = []
        for ds in DATASETS:
            v = [r[score] for r in rows if r["dataset"] == ds]
            parts.append(f"{ds}={np.mean(v):.3f}" if v else f"{ds}=n/a")
        print(f"{score:12} " + "  ".join(parts))

    # AUROC on pooled discordant pairs
    print("\n=== AUROC (curator-right-uniform-wrong vs reverse, discordant pairs pooled) ===")
    for score in ("topk_mass", "negentropy", "peak_frac"):
        ys, ss = [], []
        per_b = {}
        for b in BUDGETS:
            yb, sb = [], []
            for r in rows:
                c, u = r.get(f"cur_correct@{b}"), r.get(f"uni_correct@{b}")
                if c is None or u is None or c == u:
                    continue
                yb.append(1 if c == 1 else 0)
                sb.append(r[score])
            per_b[b] = (np.array(yb), np.array(sb))
            ys += yb
            ss += sb
        pooled = auroc(np.array(ys), np.array(ss))
        detail = "  ".join(
            f"b{b}={auroc(*per_b[b]):.3f}(n={len(per_b[b][0])})" for b in BUDGETS if len(per_b[b][0])
        )
        print(f"{score:12} pooled={pooled:.3f} (n={len(ys)})   {detail}")
    print("\nGate G1: best pooled AUROC >= 0.62 -> C3 (router) is IN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
