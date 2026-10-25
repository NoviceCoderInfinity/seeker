#!/usr/bin/env python
"""Amortized WRITE-cost reuse curve (I5, community-protocol contribution).

Video-agent papers report per-question frame counts but ignore that the memory
WRITE (scan captioning) is built ONCE per video and reused across every question
on that video. The scan ledger is deterministic and question-agnostic (P5:
video-keyed caching changes NOTHING downstream, dossier §6), so amortized WRITE
cost = W / r where W = 4.0 VLM captioning calls/video (P5 anchor) and r =
realized questions-per-video. This figure plots the analytic curve W/r plus one
measured point per held-out set at its ACTUAL reuse factor.

Output: reports/figures/fig_reuse_curve.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W = 4.0  # uncached scan VLM calls / video (P5 measured anchor, dossier §6)

# (label, reuse r) — r measured from questions-per-video in each held-out set.
PTS = [
    ("EgoSchema", 1.00),
    ("NExT-GQA", 1.09),
    ("MLVU", 1.15),
    ("NExT-QA", 1.24),
    ("NExT-GQA-360", 1.41),
    ("Video-MME", 1.65),
    ("LVBench", 3.11),
]

# policy-identity palette (matches n7_figures.py / fig_budget_collapse.py)
CDDE = "#1baf7a"
TXT1, TXT2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#d8d7d2", "axes.linewidth": 0.8,
    "axes.labelcolor": TXT1, "text.color": TXT1,
    "xtick.color": TXT2, "ytick.color": TXT2,
    "axes.grid": True, "grid.color": "#eceae5", "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "svg.fonttype": "none",
})


def main():
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    # analytic amortized-WRITE curve W/r
    rr = np.linspace(1.0, 3.3, 200)
    ax.plot(rr, W / rr, color=CDDE, linewidth=1.8, zorder=2)
    # uncached baseline (no video-keyed cache): flat W
    ax.axhline(W, color="#b9b7b0", linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)
    ax.text(3.28, W - 0.12, "uncached (per-Q rebuild)", ha="right", va="top",
            fontsize=7.2, color=TXT2)

    # measured dataset points at their realized reuse
    xs = [r for _, r in PTS]
    ys = [W / r for _, r in PTS]
    ax.scatter(xs, ys, s=34, color=CDDE, edgecolor="white", linewidth=1.1, zorder=4)
    # the r~1.0-1.4 cluster is crowded (4 points in 0.4 width) -> route their labels
    # into the empty upper-right with thin leader lines; isolated points direct-labelled.
    # lab: (label-x, label-y, ha)
    lab_pos = {
        "EgoSchema":    (1.02, 4.22, "left"),
        "NExT-GQA":     (1.78, 4.12, "left"),
        "MLVU":         (1.78, 3.80, "left"),
        "NExT-QA":      (1.78, 3.48, "left"),
        "NExT-GQA-360": (1.78, 3.16, "left"),
        "Video-MME":    (1.78, 2.60, "left"),
        "LVBench":      (2.66, 1.55, "right"),
    }
    for (lab, r), y in zip(PTS, ys):
        lx, ly, ha = lab_pos[lab]
        leader = dict(arrowstyle="-", color="#9a988f", linewidth=0.6) \
            if abs(lx - r) > 0.12 or abs(ly - y) > 0.12 else None
        ax.annotate(lab, xy=(r, y), xytext=(lx, ly), fontsize=6.8, color=TXT1,
                    ha=ha, va="center", arrowprops=leader)

    ax.set_xlabel("realized reuse  r  (questions / video)", fontsize=8.5)
    ax.set_ylabel("amortized WRITE cost\n(VLM captioning calls / question)", fontsize=8.5)
    ax.set_xlim(0.9, 3.35)
    ax.set_ylim(0.9, 4.35)
    ax.set_title("Memory WRITE amortizes with video reuse", fontsize=9, color=TXT1, pad=5)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"reports/figures/fig_reuse_curve.{ext}", dpi=200,
                    bbox_inches="tight")
    print("wrote reports/figures/fig_reuse_curve.{pdf,png}")


if __name__ == "__main__":
    main()
