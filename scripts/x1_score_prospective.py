"""X1: score LVBench results against the PREREGISTERED predictions (reports/x1_preregistration.md).

Reads runs/x1_lvbench/summary.json (or predictions.jsonl for per-category), computes per-category
Δ = acc(scripted_curator@8) − acc(uniform@8), and evaluates the two preregistered gates:
  G-rank: Spearman(predicted category ranking, observed Δ ranking) ≥ 0.70
  G-contrast: Δ(Key Information Retrieval) − Δ(Summarization) ≥ +0.08
Writes reports/x1_prospective_validation.md. NO edits to the preregistration file.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs/x1_lvbench"
QA = ROOT / "data/processed/lvbench_val.jsonl"

# Preregistered ranking (best→worst expected Δ), locked in x1_preregistration.md
PREREG_RANK = [
    "key information retrieval",
    "temporal grounding",
    "entity recognition",
    "event understanding",
    "reasoning",
    "summarization",
]


def spearman(a, b):
    n = len(a)
    ra = {v: i for i, v in enumerate(sorted(a, key=lambda x: a[x]))}
    rb = {v: i for i, v in enumerate(sorted(b, key=lambda x: b[x]))}
    keys = list(a)
    d2 = sum((ra[k] - rb[k]) ** 2 for k in keys)
    return 1 - 6 * d2 / (n * (n * n - 1))


def main() -> int:
    if not (RUN / "predictions.jsonl").exists():
        print("x1_lvbench predictions not found — eval not finished")
        return 1
    cat = {}
    for line in QA.open():
        line = line.strip()
        if line:
            d = json.loads(line)
            cat[str(d["qid"])] = (d["metadata"]["raw"]["category"], d.get("answer_index"))
    # per category, per policy@8 correctness
    hits = defaultdict(lambda: defaultdict(list))
    for line in (RUN / "predictions.jsonl").open():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("budget") != 8:
            continue
        qid = str(d["qid"])
        if qid not in cat:
            continue
        c, gold = cat[qid]
        hits[c][d["policy"]].append(int(d.get("answer_index") == gold))

    obs = {}
    lines = ["# X1 — Prospective LVBench validation (scored vs preregistration)\n"]
    lines.append("| category | n | curator@8 | uniform@8 | Δ | predicted |")
    lines.append("|---|---|---|---|---|---|")
    pred_dir = {"key information retrieval": "+", "temporal grounding": "+", "entity recognition": "+/0",
                "event understanding": "0", "reasoning": "0", "summarization": "−/0"}
    for c in PREREG_RANK:
        cu = hits[c].get("scripted_curator", [])
        un = hits[c].get("uniform", [])
        if not cu or not un:
            continue
        acu, aun = sum(cu) / len(cu), sum(un) / len(un)
        obs[c] = acu - aun
        lines.append(f"| {c} | {len(cu)} | {acu:.3f} | {aun:.3f} | {acu-aun:+.3f} | {pred_dir[c]} |")

    if len(obs) >= 4:
        pred_scores = {c: -i for i, c in enumerate(PREREG_RANK) if c in obs}
        rho = spearman(pred_scores, obs)
        contrast = obs.get("key information retrieval", 0) - obs.get("summarization", 0)
        g_rank = rho >= 0.70
        g_contrast = contrast >= 0.08
        verdict = "VALIDATED" if (g_rank and g_contrast) else (
            "PARTIAL" if g_contrast else "FALSIFIED")
        lines += [
            "\n## Gate results (preregistered)",
            f"- Spearman(predicted rank, observed Δ) = **{rho:.3f}** (gate ≥0.70: {'PASS' if g_rank else 'FAIL'})",
            f"- Δ(KIR) − Δ(Summarization) = **{contrast:+.3f}** (gate ≥+0.08: {'PASS' if g_contrast else 'FAIL'})",
            f"- **VERDICT: {verdict}** — the evidence-locality taxonomy "
            + ("predicts held-out hour-long behavior." if verdict == "VALIDATED"
               else "partially holds; report as stated." if verdict == "PARTIAL"
               else "does NOT generalize to hour scale (reported verbatim)."),
        ]
    (ROOT / "reports/x1_prospective_validation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
