"""Figure 2: specialist vs spread with a coin-flipping blind zone.

Same idea as human_calibrated_judges.py (Figure 1), but the judge does not know
which labels to get wrong. It has a set of pairs it cannot tell apart -- its
blind zone -- and answers each of those by coin flip. Outside the zone it
repeats the human's label.

    judge = human labels, except on a blind zone Z, where the winner is random.

A coin flip lands on the human's answer half the time, so a zone of size 2k
costs the judge k wrong labels on average. Every judge gets the same zone size,
so every judge has the same expected agreement with the human, 1 - k/n. What
differs is where the zone sits, from all near-ties (q = 0) to anywhere (q = 1).

Because the flips are coins rather than known errors, the proxy is equal only
in expectation, so the table also prints how far one run wanders from it.

Run it without arguments to get the paper's Figure 2:

    uv run simulation/blind_zone_judges.py

Any argument makes it a variant, written under its own file name.
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

FIGURE = "Figure 2 (sim_blindzone_main.png)"


def run_share(args) -> dict:
    """One point of the sweep: mean results over n_runs runs at spread share q."""
    (share, population, disagree, k_close, n_items, n_pairs, n_runs, seed,
     want_human, link, gold) = args
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

        # The human annotates once.
        human = (rng.random(len(delta)) < pl.win_prob(delta, link)).astype(np.int8)

        # Separate random stream for the judge.
        jrng = np.random.default_rng([seed, run, int(round(share * 1000))])
        # Zone size = 2k, because a coin flip is wrong half the time.
        zone_size = min(2 * int(round(disagree * len(delta))), len(delta))
        zone = pick_flips(jrng, da, zone_size, share, k_close)
        judge = human.copy()
        judge[zone] = jrng.integers(0, 2, size=len(zone)).astype(np.int8)

        agr_human.append(float(np.mean(judge == human)))
        acc_gold.append(float(np.mean(judge == truth)))

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
        # Spread of single-run agreement around its expectation.
        "agr_human_sd": float(np.std(agr_human, ddof=1)),
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
        disagree_help="expected share of human labels every judge gets wrong; "
                      "the blind zone is twice this",
        k_close_help="how sharply the blind zone prefers near-ties, as "
                     "halvings per unit of gap",
        with_human=False)
    a = ap.parse_args()
    cli.require_tag(a)

    # The zone is twice the error rate, so it stops fitting above 0.5.
    if not 0.0 < a.disagree < 0.5:
        raise SystemExit("--disagree must be between 0 and 0.5 "
                         "(the blind zone is twice this)")
    shares = cli.resolve_shares(a)

    cfg = {
        "judge": "blind_zone_coinflip",
        "population": a.population, "link": a.link, "gold": a.gold,
        "disagree": a.disagree, "k_close": a.k_close,
        "n_items": a.items, "n_pairs": a.pairs, "n_runs": a.runs,
        "seed": a.seed, "shares": shares,
    }
    cli.announce(cfg, FIGURE)

    jobs = [(s, a.population, a.disagree, a.k_close, a.items, a.pairs,
             a.runs, a.seed, i == 0, a.link, a.gold)
            for i, s in enumerate(shares)]
    workers = min(len(jobs), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run_share, jobs))

    res = {"config": cfg, "rho_human": rows[0]["rho_human"], "rows": rows}

    sfx = suffix(a.population, a.tag, a.link, a.gold)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"sim_blindzone_settings{sfx}.json", "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    plots.print_table(res, FIGURE)
    print("\nspread of the proxy across runs (coin flips make it equal only on "
          "average):")
    for r in rows:
        print(f"  q={r['share']:.2f}  agreement {r['agr_human']:.4f} "
              f"+- {r['agr_human_sd']:.4f}")
    plots.plot_main(res, RESULTS_DIR / f"sim_blindzone_main{sfx}.png")
    plots.plot_sweep(res, RESULTS_DIR / f"sim_blindzone_sweep{sfx}.png")


if __name__ == "__main__":
    main()
