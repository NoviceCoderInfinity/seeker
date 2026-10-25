"""N7 — paper figures 4 (mechanism) + 5 (contamination audit) + P2 cost table.

Fig 4 (mechanism of the localizable loss): per policy on nextgqa (b4/8/16, powered runs):
  (a) evidence-span hit rate — fraction of questions where >=1 selected frame lands inside
      a gold evidence span; (b) mean frames in the answer context. Shows the learned
      curator frame-starves (fewest frames, lowest hit rate) exactly where evidence is local.

Fig 5 (contamination audit): learned-curator accuracy on SEEN (qid in GRPO train) vs
  STRICT HELD-OUT slices, next to uniform on the same slices — the memorization gap made
  visible. The audit figure.

Cost table: per policy x dataset — mean frames, VLM answer+probe calls (num_retrievals as
  retrieval hops; latency as wall-clock proxy). reports/cost_table.md

All zero-GPU: reads existing predictions. v2 harness runs only; consolidated-if-present.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGD = ROOT / "reports/figures"

# fixed palette: color follows the POLICY across every figure (dataviz reference palette)
C = {"uniform": "#2a78d6", "scripted_curator": "#eda100", "learned_curator": "#4a3aa7",
     "dde_curator": "#1baf7a", "seen": "#e34948", "heldout": "#4a3aa7"}
LBL = {"uniform": "uniform", "scripted_curator": "scripted curator",
       "learned_curator": "learned curator", "dde_curator": "DdE curator"}
TXT1, TXT2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#d8d7d2", "axes.linewidth": 0.8,
    "axes.labelcolor": TXT1, "xtick.color": TXT2, "ytick.color": TXT2,
    "axes.grid": True, "grid.color": "#eceae5", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 200,
})


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_preds(run_dir: str):
    run = ROOT / run_dir
    p = run / "predictions.jsonl"
    rows = list(iter_jsonl(p)) if p.exists() else []
    if not rows:
        for shard in sorted((run / "_shards").glob("*.jsonl")):
            rows.extend(iter_jsonl(shard))
    return rows


def load_qa(path: str):
    return {str(r["qid"]): r for r in iter_jsonl(ROOT / path)}


def span_precision(sel_ts, spans) -> float:
    """Fraction of selected frames that land inside a gold evidence span (tol=0)."""
    if not sel_ts:
        return 0.0
    hits = 0
    for t in sel_ts:
        if any(sp["start"] <= float(t) <= sp["end"] for sp in spans or []):
            hits += 1
    return hits / len(sel_ts)


def fig4_mechanism():
    qa = load_qa("data/processed/nextgqa_val_450.jsonl")
    fw = json.loads((ROOT / "data/rl/firewall_split.json").read_text())
    heldout = set(fw["eval"]["nextgqa"]["heldout_qids"])
    sources = {
        "uniform": load_preds("runs/gpu_nextgqa_qwen2_5_learned_grpo_pow452"),
        "scripted_curator": load_preds("runs/gpu_nextgqa_qwen2_5_learned_grpo_pow452"),
        "learned_curator": load_preds("runs/s1181_nextgqa_452"),
    }
    budgets = [4, 8, 16]
    hit, frames = {}, {}
    for pol, rows in sources.items():
        for r in rows:
            if r["policy"] != pol or str(r["qid"]) not in heldout:
                continue
            b = int(r["budget"])
            if b not in budgets:
                continue
            q = qa.get(str(r["qid"]))
            if not q:
                continue
            hit.setdefault((pol, b), []).append(
                span_precision(r.get("selected_timestamps"), q.get("evidence_spans")))
            frames.setdefault((pol, b), []).append(float(r.get("num_frames", 0)))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    x = np.arange(len(budgets))
    w = 0.26
    for i, pol in enumerate(["uniform", "scripted_curator", "learned_curator"]):
        hr = [np.mean(hit[(pol, b)]) for b in budgets]
        se = [np.std(hit[(pol, b)]) / np.sqrt(len(hit[(pol, b)])) for b in budgets]
        fr = [np.mean(frames[(pol, b)]) for b in budgets]
        axes[0].bar(x + (i - 1) * w, hr, w * 0.92, yerr=np.array(se) * 1.96, capsize=2,
                    color=C[pol], label=LBL[pol], error_kw=dict(lw=0.8, ecolor=TXT2))
        axes[1].bar(x + (i - 1) * w, fr, w * 0.92, color=C[pol], label=LBL[pol])
        for xi, v in zip(x, fr):
            axes[1].text(xi + (i - 1) * w, v + 0.25, f"{v:.1f}", ha="center",
                         fontsize=7, color=TXT2)
    axes[0].set_ylabel("evidence precision")
    axes[0].set_title("(a) fraction of selected frames on gold evidence", fontsize=9, color=TXT1)
    axes[1].set_ylabel("frames in answer context")
    axes[1].set_title("(b) frames actually used (budget = dashed)", fontsize=9, color=TXT1)
    for xi, b in zip(x, budgets):
        axes[1].hlines(b, xi - 1.6 * w, xi + 1.6 * w, ls="--", lw=0.9, color=TXT2, zorder=1)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"b={b}" for b in budgets])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("NExT-GQA, strict held-out (n=360)", fontsize=9.5, color=TXT1, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGD / f"fig4_mechanism.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("fig4 written")


def fig5_contamination():
    fw = json.loads((ROOT / "data/rl/firewall_split.json").read_text())
    train_q = set(fw["train_qids"])
    sets = {
        "Video-MME": ("videomme", "runs/s1181_videomme_453", "runs/r2_v2_videomme_qwen2_5_full"),
        "LongVideoBench": ("longvideobench", "runs/s1181_lvb_669", "runs/r2_v2_longvideobench_qwen2_5_full"),
        "EgoSchema": ("egoschema", "runs/s1181_egoschema_252", "runs/r2_v2_egoschema_qwen2_5_full"),
    }
    budgets = [4, 8, 16, 32]
    bars = {}  # ds -> (learned_seen, learned_held, uniform_seen, uniform_held, n_seen, n_held)
    for name, (ds, lrun, urun) in sets.items():
        gold = {q: int(r["answer_index"]) for q, r in load_qa(fw["eval"][ds]["file"]).items()
                if r.get("answer_index") is not None}
        heldout = set(fw["eval"][ds]["heldout_qids"])
        acc = {}
        for pol, run in (("learned_curator", lrun), ("uniform", urun)):
            for r in load_preds(run):
                if r["policy"] != pol or int(r["budget"]) not in budgets:
                    continue
                q = str(r["qid"])
                if q not in gold:
                    continue
                grp = "seen" if q in train_q else ("held" if q in heldout else None)
                if grp is None:
                    continue
                ok = float(r.get("answer_index") is not None
                           and int(r["answer_index"]) == gold[q])
                acc.setdefault((pol, grp), []).append(ok)
        bars[name] = {k: (np.mean(v), len(v)) for k, v in acc.items()}

    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    x = np.arange(len(sets))
    w = 0.19
    layout = [("learned_curator", "seen", C["seen"], "learned — SEEN in training"),
              ("learned_curator", "held", C["heldout"], "learned — strict held-out"),
              ("uniform", "seen", "#f0b9b8", "uniform — same seen slice"),
              ("uniform", "held", "#b9c9ea", "uniform — same held-out slice")]
    for i, (pol, grp, col, lab) in enumerate(layout):
        vals = [bars[n].get((pol, grp), (np.nan, 0))[0] for n in sets]
        ax.bar(x + (i - 1.5) * w, vals, w * 0.9, color=col, label=lab)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi + (i - 1.5) * w, v + 0.012, f"{v:.2f}", ha="center",
                        fontsize=7, color=TXT2)
    ax.set_xticks(x)
    ax.set_xticklabels(list(sets))
    ax.set_ylabel("accuracy (budget-pooled)")
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.30))
    ax.set_title("Contamination audit: seen-in-training vs strict held-out",
                 fontsize=9.5, color=TXT1, pad=42)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGD / f"fig5_contamination.{ext}", bbox_inches="tight")
    plt.close(fig)
    for n, b in bars.items():
        print(n, {k: (round(v[0], 3), v[1]) for k, v in b.items()})
    print("fig5 written")


def cost_table():
    runs = {
        "nextgqa": [("uniform", "runs/gpu_nextgqa_qwen2_5_learned_grpo_pow452"),
                    ("scripted_curator", "runs/gpu_nextgqa_qwen2_5_learned_grpo_pow452"),
                    ("learned_curator", "runs/s1181_nextgqa_452")],
        "longvideobench": [("uniform", "runs/r2_v2_longvideobench_qwen2_5_full"),
                           ("scripted_curator", "runs/r2_v2_longvideobench_qwen2_5_full"),
                           ("learned_curator", "runs/s1181_lvb_669")],
        "egoschema": [("uniform", "runs/r2_v2_egoschema_qwen2_5_full"),
                      ("scripted_curator", "runs/r2_v2_egoschema_qwen2_5_full"),
                      ("learned_curator", "runs/s1181_egoschema_252")],
        "videomme": [("uniform", "runs/r2_v2_videomme_qwen2_5_full"),
                     ("scripted_curator", "runs/r2_v2_videomme_qwen2_5_full"),
                     ("learned_curator", "runs/s1181_videomme_453")],
    }
    lines = ["# P2 — end-to-end cost per policy (from run metadata, v2 harness)", "",
             "_mean over all questions x budgets present in the run; VLM calls = tentative/final",
             "answer calls (scripted probes are heuristic, learned probes are adapter calls counted",
             "separately as hops); latency includes retrieval + VLM._", "",
             "| dataset | policy | frames | retrieval hops | latency s/question |", "|---|---|---|---|---|"]
    for ds, specs in runs.items():
        for pol, run in specs:
            rows = [r for r in load_preds(run) if r["policy"] == pol]
            if not rows:
                lines.append(f"| {ds} | {LBL.get(pol, pol)} | — | — | — |")
                continue
            fr = np.mean([float(r.get("num_frames", 0)) for r in rows])
            hops = np.mean([float(r.get("num_retrievals", 0)) for r in rows])
            lat = np.mean([float(r["metadata"].get("latency_s", np.nan)) for r in rows
                           if r.get("metadata")])
            lines.append(f"| {ds} | {LBL.get(pol, pol)} | {fr:.1f} | {hops:.1f} | {lat:.1f} |")
    out = ROOT / "reports/cost_table.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"WROTE {out}")


if __name__ == "__main__":
    FIGD.mkdir(parents=True, exist_ok=True)
    fig4_mechanism()
    fig5_contamination()
    cost_table()
