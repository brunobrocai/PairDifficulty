"""The table and the two figures, shared by both simulations.

Both figures have a top panel of disagreement per gap bucket and a bottom
panel putting the proxy next to the ranking.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from . import style as fs
from .config import LABELS, NB, describe_run

spread_color = fs.spread_color


def print_table(res: dict, figure: str = "the paper figure") -> None:
    cfg = res["config"]
    print(f"\nitems={cfg['n_items']}  pairs={cfg['n_pairs']}  runs={cfg['n_runs']}  "
          f"population={cfg['population']}  human={cfg.get('human', 'bt')}  "
          f"link={cfg['link']}  "
          f"gold={cfg['gold']}  "
          f"flipped share={cfg['disagree']:.2f}  k_close={cfg['k_close']:g}")
    print(describe_run(cfg, figure))
    ref_name = ("true ability" if cfg["gold"] == "true"
                else "the human's own ranking")
    print(f"ranking scored against {ref_name}; "
          f"the human's own Spearman there: {res['rho_human']:.4f}\n")
    head = (f"{'spread q':>9} {'agr human':>10} {'acc gold':>9} {'rho':>8} "
            f"{'se':>7} {'d rho vs q=0':>13}")
    print(head)
    print("-" * len(head))
    base = res["rows"][0]["rho"]
    for r in res["rows"]:
        print(f"{r['share']:>9.2f} {r['agr_human']:>10.4f} {r['acc_gold']:>9.4f} "
              f"{r['rho']:>8.4f} {r['rho_se']:>7.4f} {r['rho'] - base:>13.4f}")

    print(f"\n{'bucket |dbeta|':>15} {'pair share':>11}", end="")
    for r in res["rows"]:
        print(f" {'q=' + format(r['share'], '.1f'):>8}", end="")
    print()
    for b in range(NB):
        print(f"{LABELS[b]:>15} {res['rows'][0]['bucket_share'][b]:>11.3f}", end="")
        for r in res["rows"]:
            print(f" {r['dis_by_bucket'][b]:>8.3f}", end="")
        print()


def arm_label(q: float) -> str:
    return f"q = {q:g}"


def rank_axis_label(gold: str) -> str:
    """What the right-hand bar group is scored against."""
    return ("Ranking correlation with true ability" if gold == "true"
            else "Ranking correlation with the human's ranking")


def draw_top_panel(ax, res) -> None:
    """Per-|Delta| disagreement with the human. The specialist (q=0) tracks the
    human's own near-tie noise; raising q lifts the curve off the near-ties and
    onto the pairs the human got right."""
    rows = res["rows"]
    x = np.arange(NB)
    spec, spreads = rows[0], rows[1:]
    for i, r in enumerate(spreads):
        ax.plot(x, r["dis_by_bucket"], marker=".", lw=1.8,
                color=spread_color(i, len(spreads)), label=arm_label(r["share"]))
    ax.plot(x, spec["dis_by_bucket"], marker="o", lw=3.2, color=fs.SPECIALIST, zorder=6,
            label=arm_label(spec["share"]))
    ax.plot(x, spec["human_by_bucket"], ls=":", lw=1.6, color=fs.HUMAN,
            label="human vs truth")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_xlabel("Pair-strength |Δ|")
    ax.set_ylabel("Disagreement with the human")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)


def plot_main(res: dict, out_path: Path) -> None:
    """Two-panel figure. Bottom-left is the proxy (equal across arms by
    construction), bottom-right is the ranking."""
    plt.rcParams.update({"font.size": 18, "axes.labelsize": 19,
                         "xtick.labelsize": 16, "ytick.labelsize": 16,
                         "legend.fontsize": 15})
    rows = res["rows"]
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(10, 12))
    draw_top_panel(axT, res)

    spreads = rows[1:]
    judges = [(arm_label(rows[0]["share"]), fs.SPECIALIST, rows[0])]
    for i, r in enumerate(spreads):
        judges.append((arm_label(r["share"]), spread_color(i, len(spreads)), r))

    nb = len(judges)
    width = 0.82 / nb
    gx = np.array([0.0, 1.25])
    for j, (label, colour, r) in enumerate(judges):
        off = (j - (nb - 1) / 2) * width
        axB.bar(gx[0] + off, r["agr_human"], width, color=colour, label=label)
        axB.text(gx[0] + off, r["agr_human"] + 0.012, f"{r['agr_human']:.2f}",
                 ha="center", fontsize=12)
        axB.bar(gx[1] + off, r["rho"], width, color=colour,
                yerr=r["rho_se"], capsize=3)
        axB.text(gx[1] + off, r["rho"] + 0.012, f"{r['rho']:.2f}",
                 ha="center", fontsize=12)
    axB.set_xticks(gx)
    axB.set_xticklabels(["Pairwise agreement with human",
                         rank_axis_label(res["config"]["gold"])])
    axB.set_ylabel("Score")
    axB.set_ylim(0.0, 1.0)
    axB.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")


def plot_sweep(res: dict, path: Path) -> None:
    """Ranking quality across the whole specialist-to-spread axis, with the
    proxy flat on top of it."""
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 15})
    rows = res["rows"]
    q = [r["share"] for r in rows]
    rho = [r["rho"] for r in rows]
    se = [r["rho_se"] for r in rows]
    agr = [r["agr_human"] for r in rows]
    acc = [r["acc_gold"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    gold = res["config"]["gold"]
    ax.errorbar(q, rho, yerr=se, marker="o", lw=2.5, capsize=4, color=fs.SPECIALIST,
                label=("Spearman vs true ability" if gold == "true"
                       else "Spearman vs the human's ranking"))
    ax.plot(q, agr, marker="s", lw=2.5, color=fs.EXTRA[1],
            label="Agreement with the human (the proxy)")
    ax.plot(q, acc, marker="^", lw=2.0, ls="--", color=fs.EXTRA[2],
            label="Accuracy vs gold")
    if gold == "true":
        ax.axhline(res["rho_human"], color=fs.HUMAN, lw=1.0, ls=":",
                   label="human's own ranking")
    ax.set_xlabel("Share of the flips placed uniformly  (0 = specialist, 1 = spread)")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Plot saved: {path}")
