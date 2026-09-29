"""Do the reliability proxies predict ranking accuracy? Spearman of each proxy
(PC all/close/far, IPP, intransitivity) against ρ, pooled over all 12
(corpus, judge) points and within each model family. Permutation p, Holm-adjusted
across the proxy rows.

    uv run proxies/proxy_predicts_rho_table.py \
        [--out-pooled tables/proxy_vs_rho_pooled.tex] \
        [--out-family tables/proxy_vs_rho_byfamily.tex]
"""

from __future__ import annotations

import argparse
import math
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ASAP = REPO / "corpora" / "judge_runs" / "asap_three_proxies"
sys.path.insert(0, str(ASAP))

import posbias_close_far_table as T  # noqa: E402
from asap_metrics import (  # noqa: E402
    load_verdicts,
    triad_intransitivity_swap,
)
from scipy.stats import spearmanr  # noqa: E402

# (record key, label, sign a valid proxy would show vs ρ; None = not theory-pinned).
PROXIES = [
    ("consistency", "Consistency, all", "+"),
    ("pc_close", "Consistency, close", "+"),
    ("pc_far", "Consistency, far", "+"),
    ("primacy", "Primacy (IPP)", None),
    ("intransitivity", "Intransitivity", "-"),
]

PERM_EXACT_MAX = 50_000  # enumerate all n! permutations below this; else sample
B_PERM = 10_000
SEED = 0


def family(judge: str) -> str:
    if judge.startswith("GPT"):
        return "GPT-5.4"
    if judge.startswith("Ministral"):
        return "Ministral-3"
    if judge.startswith("Gemma"):
        return "Gemma-3"
    return "other"


# --- per-judge rows ---

def build_points() -> list[dict]:
    gold_cache: dict[str, dict[str, float]] = {}
    edge_cache: dict[str, np.ndarray] = {}
    out = []
    for corpus, key, cache_path, name in T.ROWS:
        if key not in gold_cache:
            gold_cache[key] = T.GOLD_LOADERS[key]()
        gold = gold_cache[key]
        if key not in edge_cache:
            edge_cache[key] = T._gap_quartile_edges(sorted(gold), gold)
        # PC/IPP/ρ from the posbias table's compute_row.
        prow = T.compute_row(cache_path, gold, edge_cache[key])

        rows = load_verdicts(cache_path)
        uids = sorted(({r["uid_a"] for r in rows} | {r["uid_b"] for r in rows})
                      & set(gold))
        ti = triad_intransitivity_swap(rows, uids)
        ntri = ti["n_triads"]
        intrans = ((ti["n_cycle"] + ti["n_mixed"] + ti["n_equality_chain"]) / ntri
                   if ntri else float("nan"))

        out.append({
            "corpus": corpus, "judge": name, "family": family(name),
            "consistency": prow["pc_all"],    # PC overall
            "pc_close": prow["pc_close"],      # PC, close (Q1 gold gap)
            "pc_far": prow["pc_far"],          # PC, far (Q4 gold gap)
            "primacy": prow["ipp_all"],        # IPP
            "intransitivity": intrans,         # total % triads
            "rho": prow["rho"],
        })
    return out


# --- Spearman ---

def spearman_perm(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman r of x vs y plus a two-sided permutation p on |r|.

    Exact (enumerate every relabelling of y) when n! <= PERM_EXACT_MAX, else
    B_PERM random shuffles. Returns r, p, n."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = len(x)
    if n < 3:
        return {"r": float("nan"), "p": float("nan"), "n": n}
    r = float(spearmanr(x, y).statistic)
    thresh = abs(r) - 1e-12
    if math.factorial(n) <= PERM_EXACT_MAX:
        idx = list(permutations(range(n)))
        rs = np.array([spearmanr(x, y[list(p)]).statistic for p in idx])
        p = float(np.mean(np.abs(rs) >= thresh))
        exact = True
    else:
        rng = np.random.default_rng(SEED)
        ge = 0
        for _ in range(B_PERM):
            ge += abs(spearmanr(x, rng.permutation(y)).statistic) >= thresh
        p = (ge + 1) / (B_PERM + 1)
        exact = False
    return {"r": r, "p": p, "n": n, "exact": exact}


def _holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values; preserves input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def correlate(points: list[dict]) -> dict:
    """{proxy_key -> spearman_perm result} with a Holm-adjusted `p_holm` across
    the proxies (a family of correlations against the same ρ)."""
    rho = np.array([pt["rho"] for pt in points], float)
    res = {key: spearman_perm(np.array([pt[key] for pt in points], float), rho)
           for key, _label, _sign in PROXIES}
    keys = [key for key, _l, _s in PROXIES]
    raw = [res[k]["p"] for k in keys]
    safe = [p if p == p else 1.0 for p in raw]   # NaN p -> 1.0 for the step-down
    for k, p0, padj in zip(keys, raw, _holm(safe)):
        res[k]["p_holm"] = padj if p0 == p0 else float("nan")
    return res


# --- Rendering ---

def _stars(p: float) -> str:
    if p != p:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def _fmt_p(p: float) -> str:
    if p != p:
        return "--"
    if p < 1e-3:
        return r"$<\!10^{-3}$"
    return f"${p:.3f}$"


def to_latex_pooled(res: dict) -> str:
    lines = [
        r"\begin{tabular}{lcc}",
        r"  \toprule",
        r"  Proxy & $\rho_s$ vs ranking $\rho$ & $p_{\text{Holm}}$ \\",
        r"  \midrule",
    ]
    for key, label, _sign in PROXIES:
        r = res[key]
        lines.append(f"  {label:<28} & ${r['r']:+.3f}$ & {_fmt_p(r['p_holm'])} \\\\")
    lines += [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def to_latex_family(by_fam: dict, fams: list[str], n_per: dict) -> str:
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{f} ($n{{=}}{n_per[f]}$)}}"
                       for f in fams)
    cmid = "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
                   for i in range(len(fams)))
    subhead = " & ".join(r"$\rho_s$ & $p_{\text{Holm}}$" for _ in fams)
    lines = [
        rf"\begin{{tabular}}{{l{'cc' * len(fams)}}}",
        r"  \toprule",
        rf"   & {head} \\",
        f"  {cmid}",
        rf"  Proxy & {subhead} \\",
        r"  \midrule",
    ]
    for key, label, _sign in PROXIES:
        cells = []
        for f in fams:
            r = by_fam[f][key]
            cells.append(f"${r['r']:+.3f}$ & {_fmt_p(r['p_holm'])}")
        lines.append(f"  {label:<28} & " + " & ".join(cells) + r" \\")
    lines += [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def text_preview(points: list[dict], pooled: dict, by_fam: dict,
                 fams: list[str], n_per: dict) -> str:
    out = ["per-point proxy values (each = one corpus x judge):",
           f"{'corpus':<7}{'judge':<16}{'PC':>7}{'PCcl':>7}{'PCfar':>7}"
           f"{'IPP':>7}{'intr%':>7}{'ρ':>8}",
           "-" * 61]
    for pt in points:
        out.append(f"{pt['corpus']:<7}{pt['judge']:<16}"
                   f"{pt['consistency']:>7.3f}{pt['pc_close']:>7.3f}{pt['pc_far']:>7.3f}"
                   f"{pt['primacy']:>7.3f}{100 * pt['intransitivity']:>7.1f}"
                   f"{pt['rho']:>+8.3f}")

    out += ["", f"POOLED  Spearman(proxy, ρ) across all {pooled['consistency']['n']} "
                f"points  (p_Holm across the {len(PROXIES)} proxies):"]
    for key, label, sign in PROXIES:
        r = pooled[key]
        want = f"valid:{sign}" if sign else "sign n/a"
        out.append(f"  {label:<24} r={r['r']:+.3f}  p={r['p']:.4f}  "
                   f"p_Holm={r['p_holm']:.4f} {_stars(r['p_holm'])}  ({want})")

    out += ["", "WITHIN FAMILY  Spearman(proxy, ρ), p_Holm  [low n — read sign/magnitude]:"]
    hdr = f"  {'proxy':<24}" + "".join(f"{f + f'(n={n_per[f]})':>20}" for f in fams)
    out.append(hdr)
    for key, label, _sign in PROXIES:
        cells = "".join(
            f"{f'r={by_fam[f][key]['r']:+.2f} pH={by_fam[f][key]['p_holm']:.2f}':>20}"
            for f in fams)
        out.append(f"  {label:<24}{cells}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-pooled", type=Path, default=None,
                    help="write the pooled (all points) .tex here")
    ap.add_argument("--out-family", type=Path, default=None,
                    help="write the within-family .tex here")
    args = ap.parse_args(argv)

    points = build_points()
    pooled = correlate(points)

    fams = list(dict.fromkeys(pt["family"] for pt in points))
    by_fam = {f: correlate([pt for pt in points if pt["family"] == f]) for f in fams}
    n_per = {f: sum(pt["family"] == f for pt in points) for f in fams}

    print(text_preview(points, pooled, by_fam, fams, n_per))
    print("\n% === pooled (all corpus x judge points) ===")
    tex_pooled = to_latex_pooled(pooled)
    print(tex_pooled)
    print("\n% === within model family ===")
    tex_family = to_latex_family(by_fam, fams, n_per)
    print(tex_family)

    for out, tex in ((args.out_pooled, tex_pooled), (args.out_family, tex_family)):
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(tex + "\n")
            print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
