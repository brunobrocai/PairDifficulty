"""Ties-excluded robustness checks for the ASAP close/far result.

All numbers are computed from the existing verdict caches. For ASAP, "close" is
redefined as gold gap == 1 and "far" as gap >= 2, so the gold-tied (gap 0) pairs
that otherwise fill the close quartile are dropped.

  1. Tie share: % of the sampled (judged) ASAP unordered pairs with gap 0.
  2. Consistency (PC = 1 - flip) close(gap1) vs far(gap>=2), per ASAP judge.
  3. rho vs gold with gold-tied pairs removed from the BT input, per judge,
     next to the baseline rho (all verdicts).
  4. Table-2 pooled Spearman(PC-far, rho) recomputed under the new binning.
  5. Triad intransitivity counts with gold-tied pairs excluded.

    uv run proxies/asap_ties_excluded_check.py [--out tables/asap_ties_excluded.tex]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ASAP = REPO / "corpora" / "judge_runs" / "asap_three_proxies"
sys.path.insert(0, str(ASAP))

import posbias_close_far_table as T  # noqa: E402
from asap_metrics import (  # noqa: E402
    _unordered_pair_key,
    bt_fit,
    load_verdicts,
    spearman,
    triad_intransitivity_swap,
)
from scipy.stats import spearmanr  # noqa: E402

ASAP_ROWS = [r for r in T.ROWS if r[0] == "ASAP"]


def _uids_and_gold(cache_path: Path, gold: dict[str, float]):
    rows = load_verdicts(cache_path)
    uids = sorted(({r["uid_a"] for r in rows} | {r["uid_b"] for r in rows})
                  & set(gold))
    return rows, uids


# --- 1. tie share among the judged pairs ---

def tie_share(gold: dict[str, float]) -> dict:
    """Fraction of distinct judged unordered pairs (pooled over ASAP judges)
    whose gold gap is 0. Counts each unordered pair once across all judges."""
    seen: set[tuple[str, str]] = set()
    tied: set[tuple[str, str]] = set()
    for _corpus, _key, cache_path, _name in ASAP_ROWS:
        for r in load_verdicts(cache_path):
            a, b = r.get("uid_a"), r.get("uid_b")
            if a not in gold or b not in gold:
                continue
            key = _unordered_pair_key(a, b)
            seen.add(key)
            if gold[a] == gold[b]:
                tied.add(key)
    n, n_tie = len(seen), len(tied)
    return {"n_pairs": n, "n_tied": n_tie,
            "share": n_tie / n if n else float("nan")}


# --- 3. rho vs gold with gold-tied verdicts removed from the BT input ---

def rho_ties_excluded(cache_path: Path, gold: dict[str, float]) -> dict:
    rows, uids = _uids_and_gold(cache_path, gold)
    g_arr = np.array([gold[u] for u in uids], float)
    rho_all = spearman(bt_fit(uids, rows, alpha=T.ALPHA), g_arr)
    kept = [r for r in rows
            if r.get("uid_a") in gold and r.get("uid_b") in gold
            and gold[r["uid_a"]] != gold[r["uid_b"]]]
    rho_no_ties = spearman(bt_fit(uids, kept, alpha=T.ALPHA), g_arr)
    return {"rho_all": rho_all, "rho_no_ties": rho_no_ties,
            "delta": rho_no_ties - rho_all,
            "n_verdicts_all": len(rows), "n_verdicts_kept": len(kept)}


# --- 5. intransitivity counts with gold-tied pairs excluded ---

def _rows_without_tied_pairs(rows, gold):
    return [r for r in rows
            if r.get("uid_a") in gold and r.get("uid_b") in gold
            and gold[r["uid_a"]] != gold[r["uid_b"]]]


def intransitivity_rebinned(cache_path: Path, gold: dict[str, float]) -> dict:
    rows, uids = _uids_and_gold(cache_path, gold)
    full = triad_intransitivity_swap(rows, uids)
    kept = triad_intransitivity_swap(_rows_without_tied_pairs(rows, gold), uids)
    return {"full": full, "no_tied_pairs": kept}


# --- appendix table: baseline vs ties-excluded, ASAP only ---

def build_records(gold: dict[str, float], edges) -> list[dict]:
    """One record per ASAP judge holding both binnings side by side.

    `close0` is the baseline Table-1 close bucket, which on ASAP is the gold-tied
    pairs. `close1` is the rebinned close bucket, gold gap exactly 1. `far` is
    gap >= 2 under both binnings, so it is stored once.
    """
    out = []
    for _c, _k, cache_path, name in ASAP_ROWS:
        base = T.compute_row(cache_path, gold, edges)
        reb = T.compute_row(cache_path, gold, edges, integer_diff_buckets=True)
        rx = rho_ties_excluded(cache_path, gold)
        # On ASAP the Q4 edge is 2.0, so both far buckets are the same pairs.
        assert abs(base["pc_far"] - reb["pc_far"]) < 1e-12, name
        out.append({
            "judge": name,
            "close0": base["pc_close"], "close1": reb["pc_close"],
            "far": reb["pc_far"],
            "n_close0": base["nb_close"], "n_close1": reb["nb_close"],
            "n_far": reb["nb_far"], "n_all": base["nb_all"],
            "rho_all": rx["rho_all"], "rho_no_ties": rx["rho_no_ties"],
        })
    return out


def to_latex(recs: list[dict]) -> str:
    """Bare booktabs tabular; written by hand because of the two-level header.

    close (gap 0) is Table 1's ASAP close bucket (the gold-tied pairs);
    close (gap 1) is the rebinned one. far is the same pairs under both.
    Consistency = 1 - order-swap flip rate. Bucket sizes are shares of all
    judged pairs. rho = Spearman between the judge's Bradley-Terry ranking and
    the gold, fitted on all verdicts and then with the gold-tied ones dropped.
    """
    n = recs[0]
    assert all(r["n_close0"] == n["n_close0"] and r["n_close1"] == n["n_close1"]
               and r["n_far"] == n["n_far"] and r["n_all"] == n["n_all"]
               for r in recs), "bucket n varies"
    pct = lambda k: 100 * n[k] / n["n_all"]  # noqa: E731
    lines = [
        r"\begin{tabular}{lccccc}",
        r"  \toprule",
        r"  & \multicolumn{3}{c}{Consistency ($1-$flip)} & "
        r"\multicolumn{2}{c}{$\rho$ vs gold} \\",
        r"  \cmidrule(lr){2-4}\cmidrule(lr){5-6}",
        r"  Judge & close & close & far & all & tied \\",
        r"  & (gap $=0$) & (gap $=1$) & (gap $\geq 2$) & verdicts & excluded \\",
        rf"  & {pct('n_close0'):.1f}\% & {pct('n_close1'):.1f}\% "
        rf"& {pct('n_far'):.1f}\% & & \\",
        r"  \midrule",
    ]
    for r in recs:
        lines.append(
            f"  {r['judge']:<16} & "
            f"{T._fmt2(r['close0'])} & {T._fmt2(r['close1'])} & {T._fmt2(r['far'])} & "
            f"${r['rho_all']:+.3f}$ & ${r['rho_no_ties']:+.3f}$ \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="write the appendix .tex here (also printed)")
    args = ap.parse_args(argv)

    gold = T._gold_asap()
    edges = T._gap_quartile_edges(sorted(gold), gold)

    # 1 -----------------------------------------------------------------
    ts = tie_share(gold)
    print("[1] ASAP tie share (gold gap 0) among sampled unordered pairs")
    print(f"    {100 * ts['share']:.1f}%  ({ts['n_tied']}/{ts['n_pairs']} pairs)")

    # 2 + 3 -------------------------------------------------------------
    print("\n[2] PC (1-flip) close(gap1) vs far(gap>=2), per ASAP judge")
    print("[3] rho vs gold: all verdicts  ->  gold-tied pairs excluded from BT")
    hdr = (f"    {'judge':<16}{'PC.close':>9}{'PC.far':>9}"
           f"{'rho.all':>10}{'rho.noTie':>11}{'delta':>8}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for _c, _k, cache_path, name in ASAP_ROWS:
        rec = T.compute_row(cache_path, gold, edges, integer_diff_buckets=True)
        rx = rho_ties_excluded(cache_path, gold)
        print(f"    {name:<16}{rec['pc_close']:>9.2f}{rec['pc_far']:>9.2f}"
              f"{rx['rho_all']:>+10.3f}{rx['rho_no_ties']:>+11.3f}"
              f"{rx['delta']:>+8.3f}")

    # 4 -----------------------------------------------------------------
    print("\n[4] Pooled Spearman(PC-far, rho) over all 12 points, new binning")
    pc_far, rho = [], []
    gold_cache = {"asap": gold}
    edge_cache = {"asap": edges}
    for corpus, key, cache_path, _name in T.ROWS:
        if key not in gold_cache:
            gold_cache[key] = T.GOLD_LOADERS[key]()
        if key not in edge_cache:
            edge_cache[key] = T._gap_quartile_edges(sorted(gold_cache[key]),
                                                    gold_cache[key])
        rec = T.compute_row(cache_path, gold_cache[key], edge_cache[key],
                            integer_diff_buckets=(corpus == "ASAP"))
        pc_far.append(rec["pc_far"])
        rho.append(rec["rho"])
    r = spearmanr(pc_far, rho).statistic
    print(f"    r = {r:+.3f}   (baseline headline = +0.897)")

    # 5 -----------------------------------------------------------------
    print("\n[5] triad intransitivity, gold-tied pairs excluded")
    hdr2 = (f"    {'judge':<16}"
            f"{'rate.full':>11}{'rate.noTie':>12}"
            f"{'nTri.full':>11}{'nTri.noTie':>12}")
    print(hdr2)
    print("    " + "-" * (len(hdr2) - 4))
    for _c, _k, cache_path, name in ASAP_ROWS:
        it = intransitivity_rebinned(cache_path, gold)
        f, k = it["full"], it["no_tied_pairs"]
        print(f"    {name:<16}"
              f"{f['rate']:>11.3f}{k['rate']:>12.3f}"
              f"{f['n_triads']:>11d}{k['n_triads']:>12d}")

    # appendix table -----------------------------------------------------
    tex = to_latex(build_records(gold, edges))
    print("\n[table] appendix robustness table (ASAP)\n")
    print(tex)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(tex + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
