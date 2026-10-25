"""X5 (plan4 §3.7): is the localizable/global type-level boundary selector-agnostic?

Recomputes the per-question-type net-gain table (as in reports/router.md) for THREE selectors
(scripted_curator, aks, videotree) vs uniform, from the r1_v2 ablation runs (all policies share
one config/answerer, so parity is by construction). If the same types gain/lose for aks and
videotree as for the curator, the boundary is a property of retrieval-based selection in general.

net = #budget-cells selector correct − #budget-cells uniform correct, per qid, budgets {2,4,8,16,32}.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUDGETS = [2, 4, 8, 16, 32]
SELECTORS = ["scripted_curator", "aks", "videotree"]

DATASETS = {
    "nextgqa": dict(qa="data/processed/nextgqa_val_siglip.jsonl", run="runs/r1_v2_nextgqa_qwen2_5",
                    tkey=lambda d, raw: d.get("type") or (d.get("metadata") or {}).get("type") or raw.get("type")),
    "nextqa": dict(qa="data/processed/nextqa_temporal_val_150_siglip.jsonl", run="runs/r1_v2_nextqa_qwen2_5",
                   tkey=lambda d, raw: d.get("type") or (d.get("metadata") or {}).get("type") or raw.get("type")),
    "mlvu": dict(qa="data/processed/mlvu_val_siglip.jsonl", run="runs/r1_v2_mlvu_qwen2_5",
                 tkey=lambda d, raw: raw.get("question_type")),
    "videomme": dict(qa="data/processed/videomme_long_siglip.jsonl", run="runs/r1_v2_videomme_qwen2_5",
                     tkey=lambda d, raw: raw.get("task_type") or d.get("task_type")),
    "longvideobench": dict(qa="data/processed/longvideobench_val_siglip.jsonl",
                           run="runs/r1_v2_longvideobench_qwen2_5",
                           tkey=lambda d, raw: raw.get("question_category")),
    "egoschema": dict(qa="data/processed/egoschema_val_siglip.jsonl", run="runs/r1_v2_egoschema_qwen2_5",
                      tkey=lambda d, raw: "egoschema_all"),
}


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_predictions(run_dir: Path):
    preds = defaultdict(dict)
    merged = run_dir / "predictions.jsonl"
    files = [merged] if merged.exists() else sorted((run_dir / "_shards").glob("b*_*.jsonl"))
    for path in files:
        for d in iter_jsonl(path):
            preds[str(d["qid"])][(d["policy"], d["budget"])] = d.get("answer_index")
    return preds


def main() -> int:
    table = defaultdict(lambda: defaultdict(list))  # (dataset,type) -> selector -> [net]
    for ds, spec in DATASETS.items():
        run_dir = ROOT / spec["run"]
        if not (run_dir / "predictions.jsonl").exists() and not (run_dir / "_shards").exists():
            print(f"{ds}: run missing, skipped")
            continue
        preds = load_predictions(run_dir)
        gold, qtype = {}, {}
        for d in iter_jsonl(ROOT / spec["qa"]):
            qid = str(d["qid"])
            if qid in preds:
                gold[qid] = d.get("answer_index")
                raw = (d.get("metadata") or {}).get("raw") or {}
                qtype[qid] = spec["tkey"](d, raw) or "?"
        for qid, pol_preds in preds.items():
            g = gold.get(qid)
            if g is None:
                continue
            for sel in SELECTORS:
                net, used = 0, 0
                for b in BUDGETS:
                    s = pol_preds.get((sel, b))
                    u = pol_preds.get(("uniform", b))
                    if s is None and u is None:
                        continue
                    net += int(s == g) - int(u == g)
                    used += 1
                if used:
                    table[(ds, qtype[qid])][sel].append(net)

    print(f"{'dataset':16} {'type':24} {'n':>4} " + "".join(f"{s:>18}" for s in SELECTORS))
    out = []
    for (ds, t), sels in sorted(table.items(), key=lambda kv: (kv[0][0], -np.mean(kv[1].get('scripted_curator', [0])))):
        n = len(sels.get("scripted_curator", []))
        if n < 10:
            continue
        row = {"dataset": ds, "type": t, "n": n}
        line = f"{ds:16} {str(t):24} {n:>4} "
        for s in SELECTORS:
            v = sels.get(s, [])
            m = float(np.mean(v)) if v else float("nan")
            row[s] = m
            line += f"{m:>+18.3f}"
        out.append(row)
        print(line)

    # agreement: sign correlation between curator net and aks/videotree net across type cells
    cur = np.array([r["scripted_curator"] for r in out])
    for s in SELECTORS[1:]:
        other = np.array([r[s] for r in out])
        ok = ~(np.isnan(cur) | np.isnan(other))
        if ok.sum() > 3:
            r = float(np.corrcoef(cur[ok], other[ok])[0, 1])
            sign = float(((cur[ok] > 0.1) == (other[ok] > 0.1)).mean())
            print(f"\n{s}: type-cell correlation with curator gains r={r:.3f}, direction agreement={sign:.0%}")
    Path(ROOT / "reports/x5_type_gains_by_selector.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
