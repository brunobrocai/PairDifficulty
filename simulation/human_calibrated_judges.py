"""Figure 1: specialist vs spread, calibrated on the human labels.

The human annotates first. A judge is then defined directly against those
labels:

    judge = the human labels, with exactly k of them flipped.

Every judge flips the same number k, so agreement with the human is exactly
1 - k/n for all of them, in every run. Judges differ only in which k labels
they flip:

    spread share q = 0   all k flips are taken from the closest pairs
    spread share q = 1   all k flips are picked uniformly at random, so they
                         land on far pairs as readily as on near-ties

Run it without arguments to get the paper's Figure 1:

    uv run simulation/human_calibrated_judges.py

Any argument makes it a variant, written under its own file name. The appendix
robustness checks are:

    uv run simulation/human_calibrated_judges.py --population uniform
    uv run simulation/human_calibrated_judges.py --human deterministic
    uv run simulation/human_calibrated_judges.py --disagree 0.2 --tag agr80

"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import spearmanr

from btsim import cli, links as pl, plots
from btsim.bt import build_wins_matrix, fit_bt_mm
from btsim.config import BUCKETS, NB, RESULTS_DIR
from btsim.naming import suffix
from btsim.population import draw_beta, pick_flips

FIGURE = "Figure 1 (sim_humancalib_main.png)"


def run_share(args) -> dict:
    """One point of the sweep: mean results over n_runs runs at spread share q."""
    (share, population, disagree, k_close, n_items, n_pairs, n_runs, seed,
     want_human, link, gold, human_model) = args
    all_i, all_j = np.triu_indices(n_items, k=1)
    n_all = len(all_i)

    rho, agr_human, acc_gold, rho_human = [], [], [], []
    dis_sum = np.zeros(NB)      # disagreement with the human, per |dbeta| bucket
    hum_sum = np.zeros(NB)      # the human's own disagreement with gold
    cnt_sum = np.zeros(NB)

    for run in range(n_runs):
        # Seeded by run only, so every arm sees the same population, pairs
        # and human labels.
        rng = np.random.default_rng([seed, run])
        beta = draw_beta(rng, population, n_items)

        sel = rng.choice(n_all, size=min(n_pairs, n_all), replace=False)
        pi, pj = all_i[sel], all_j[sel]
        delta = beta[pi] - beta[pj]
        da = np.abs(delta)
        truth = (delta > 0).astype(np.int8)

        # The human annotates once. The random draw is used either way, so
        # both human models see the same stream.
        u = rng.random(len(delta))
        human = (truth if human_model == "deterministic"
                 else (u < pl.win_prob(delta, link)).astype(np.int8))

        # Separate random stream for the judge.
        jrng = np.random.default_rng([seed, run, int(round(share * 1000))])
        k = int(round(disagree * len(delta)))
        flips = pick_flips(jrng, da, k, share, k_close)
        judge = human.copy()
        judge[flips] = 1 - judge[flips]

        agr_human.append(float(np.mean(judge == human)))
        acc_gold.append(float(np.mean(judge == truth)))

        # With gold="human" the reference is BT fitted to the human's labels.
        th_h = (fit_bt_mm(build_wins_matrix(n_items, pi, pj, human))
                if (gold == "human" or want_human) else None)
        ref = beta if gold == "true" else th_h

        theta = fit_bt_mm(build_wins_matrix(n_items, pi, pj, judge))
        rho.append(spearmanr(theta, ref).correlation)
        if want_human:
            rho_human.append(spearmanr(th_h, ref).correlation)

        b = np.clip(np.digitize(da, BUCKETS) - 1, 0, NB - 1)
        np.add.at(dis_sum, b, (judge != human).astype(float))
        np.add.at(hum_sum, b, (human != truth).astype(float))
        np.add.at(cnt_sum, b, 1.0)

    return {
        "share": share,
        "agr_human": float(np.mean(agr_human)),
        "acc_gold": float(np.mean(acc_gold)),
        "rho": float(np.mean(rho)),
        "rho_se": float(np.std(rho, ddof=1) / np.sqrt(n_runs)),
        "rho_human": float(np.mean(rho_human)) if rho_human else None,
        "dis_by_bucket": (dis_sum / np.maximum(cnt_sum, 1.0)).tolist(),
        "human_by_bucket": (hum_sum / np.maximum(cnt_sum, 1.0)).tolist(),
        "bucket_share": (cnt_sum / cnt_sum.sum()).tolist(),
    }


def main() -> None:
    ap = cli.build_parser(
        description=__doc__,
        disagree_help="share of human labels every judge flips",
        k_close_help="how sharply the close flips prefer near-ties, as "
                     "halvings per unit of gap: a pair's weight halves "
                     "every 1/k_close of |dbeta|",
        with_human=True)
    a = ap.parse_args()
    cli.require_tag(a)

    # Agreement with the human is 1 - the flipped share, so this keeps every
    # judge at 50% agreement or better. Past that the judge is an inverted
    # copy of the human and its ranking correlation turns negative.
    if not 0.0 < a.disagree <= 0.5:
        raise SystemExit("--disagree must be between 0 and 0.5, so agreement "
                         "with the human stays at 0.5 or above")
    shares = cli.resolve_shares(a)

    # Built before the run so the banner can say what is about to happen; the
    # same dict is what gets saved, so the two can never disagree.
    cfg = {
        "population": a.population, "human": a.human,
        "link": a.link, "gold": a.gold,
        "disagree": a.disagree,
        "k_close": a.k_close, "n_items": a.items, "n_pairs": a.pairs,
        "n_runs": a.runs, "seed": a.seed, "shares": shares,
    }
    cli.announce(cfg, FIGURE)

    jobs = [(s, a.population, a.disagree, a.k_close, a.items, a.pairs,
             a.runs, a.seed, i == 0, a.link, a.gold, a.human)
            for i, s in enumerate(shares)]
    workers = min(len(jobs), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run_share, jobs))

    res = {"config": cfg, "rho_human": rows[0]["rho_human"], "rows": rows}

    sfx = suffix(a.population, a.tag, a.link, a.gold, a.human)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"sim_humancalib_settings{sfx}.json", "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    plots.print_table(res, FIGURE)
    plots.plot_main(res, RESULTS_DIR / f"sim_humancalib_main{sfx}.png")
    plots.plot_sweep(res, RESULTS_DIR / f"sim_humancalib_sweep{sfx}.png")


if __name__ == "__main__":
    main()
