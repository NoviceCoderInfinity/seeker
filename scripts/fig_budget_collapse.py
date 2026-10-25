"""Budget-fragility figure: accuracy vs frame budget K at full n, uniform vs DdE.

Four small multiples (NExT-GQA 1680, NExT-QA 1751, EgoSchema 500, MLVU 502), shared
y-axis, K in {4,8,16}. Shows the paper's C2 headline: uniform collapses at K=16 on the
three sets where small K already covers the evidence; DdE is flat; MLVU is the
mechanism-confirming boundary case (info-starved baseline gains from frames).

Numbers frozen from reports/sigpack_v3.md (full-set panels). Zero GPU.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGD = ROOT / "reports/figures"

# fixed palette: color follows the POLICY across every figure (dataviz reference palette)
C = {"uniform": "#2a78d6", "dde_curator": "#1baf7a"}
TXT1, TXT2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#d8d7d2", "axes.linewidth": 0.8,
    "axes.labelcolor": TXT1, "text.color": TXT1,
    "xtick.color": TXT2, "ytick.color": TXT2,
    "axes.grid": True, "grid.color": "#eceae5", "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

K = [4, 8, 16, 32]
PANELS = [  # title, n, chance, uniform accs, dde accs, note
    ("NExT-GQA", 1680, 0.20, [0.649, 0.596, 0.397, 0.540], [0.667, 0.660, 0.665, 0.660], "$\\Delta_{32}$=+0.120*"),
    ("NExT-QA", 1751, 0.20, [0.683, 0.642, 0.460, 0.589], [0.685, 0.686, 0.692, 0.692], "$\\Delta_{32}$=+0.102*"),
    ("EgoSchema", 500, 0.20, [0.538, 0.384, 0.216, 0.300], [0.580, 0.514, 0.512, 0.514], "$\\Delta_{32}$=+0.214*"),
    ("MLVU", 502, 0.25, [0.291, 0.295, 0.315, 0.363], [0.365, 0.347, 0.347, 0.357], "no collapse\n(boundary case)"),
]


def main() -> None:
    FIGD.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1), sharey=True)
    for i, (ax, (title, n, chance, uni, dde)) in enumerate(zip(axes, [p[:5] for p in PANELS])):
        note = PANELS[i][5]
        Kp = K[:len(uni)]
        ax.axhline(chance, color="#b9b7b0", linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
        ax.plot(Kp, uni, color=C["uniform"], linewidth=2, marker="o", markersize=4.5,
                markerfacecolor="white", markeredgewidth=1.4, zorder=3)
        ax.plot(Kp, dde, color=C["dde_curator"], linewidth=2, marker="s", markersize=4.5,
                markerfacecolor="white", markeredgewidth=1.4, zorder=3)
        ax.set_title(f"{title} (n={n})", fontsize=9, color=TXT1, pad=4)
        ax.set_xscale("log", base=2)
        ax.set_xticks(Kp)
        ax.set_xticklabels([str(k) for k in Kp])
        ax.minorticks_off()
        ax.set_xlim(3.2, 40)
        ax.set_ylim(0.10, 0.78)
        ax.tick_params(length=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if i == 0:
            ax.set_ylabel("accuracy")
            ax.annotate("DdE (ours)", (K[-1], dde[-1]), xytext=(0, 7),
                        textcoords="offset points", ha="right", fontsize=8, color=TXT1)
            ax.annotate("uniform", (K[-1], uni[-1]), xytext=(0, -13),
                        textcoords="offset points", ha="right", fontsize=8, color=TXT1)
            ax.annotate("chance", (3.4, chance), xytext=(0, 2), textcoords="offset points",
                        fontsize=7, color=TXT2)
        if i < 3:
            ax.annotate(note, (3.5, 0.125), ha="left", va="bottom", fontsize=7.5, color=TXT2)
        else:
            ax.annotate(note, (7.5, 0.125), ha="center", va="bottom", fontsize=7.5, color=TXT2)
        ax.set_xlabel("frame budget $K$")
    handles = [
        plt.Line2D([], [], color=C["uniform"], marker="o", markersize=4.5, linewidth=2,
                   markerfacecolor="white", markeredgewidth=1.4, label="uniform"),
        plt.Line2D([], [], color=C["dde_curator"], marker="s", markersize=4.5, linewidth=2,
                   markerfacecolor="white", markeredgewidth=1.4, label="DdE (ours)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.13), fontsize=8)
    fig.tight_layout(w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(FIGD / f"fig_budget_collapse.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {FIGD}/fig_budget_collapse.{{pdf,png}}")


if __name__ == "__main__":
    main()
