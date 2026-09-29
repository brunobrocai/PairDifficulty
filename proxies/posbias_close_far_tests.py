"""Paired significance tests for the position-bias close/far table. The judges
share one schedule, so every comparison is paired at the item level: flip / IPP /
intransitivity are exact McNemar, Δρ is a paired item bootstrap. Per-corpus
within-family pairs are Holm-adjusted; the pooled small→big aggregate stratifies
across the 6 (family × corpus) steps. `--scope cross|both` widens the pair set.

Prints a text report and writes a markdown summary table (Δ and p per metric).

    uv run proxies/posbias_close_far_tests.py [--out tables/posbias_close_far_tests.md]
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import posbias_close_far_table as T  # noqa: E402
from asap_metrics import (  # noqa: E402
    bootstrap_spearman_diff,
    bt_fit,
    intransitivity_by_triad,
    load_verdicts,
    spearman,
)

B_BOOT = 10_000
SEED = 0


def family(judge: str) -> str:
    if judge.startswith("GPT"):
        return "gpt-5.4"
    if judge.startswith("Ministral"):
        return "Ministral-3"
    if judge.startswith("Gemma"):
        return "Gemma-3"
    return "other"


# --- Per-judge paired views ---

def _dir_winners(rows) -> dict[tuple[str, str], dict[str, str]]:
    """unordered pair -> {uid_in_slot_A: majority winner}, for every pair seen
    in both slot orders."""
    by_pair: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        a, b, v = r.get("uid_a"), r.get("uid_b"), r.get("verdict")
        if a is None or b is None or v not in ("A", "B"):
            continue
        key = (a, b) if a <= b else (b, a)
        winner = a if v == "A" else b
        by_pair[key][a].append(winner)  # keyed by which uid sat in slot A
    return {
        key: {first: Counter(ws).most_common(1)[0][0] for first, ws in by_first.items()}
        for key, by_first in by_pair.items()
        if len(by_first) >= 2  # only pairs seen in both slot orders
    }


def flip_by_pair(rows) -> dict[tuple[str, str], int]:
    """unordered pair -> 1 if its majority winner flips under swap."""
    return {
        key: int(len(set(dw.values())) > 1) for key, dw in _dir_winners(rows).items()
    }


def ipp_by_pair(rows) -> dict[tuple[str, str], int]:
    """unordered pair -> 1 if it flips toward primacy, 0 toward recency. Only
    pairs that flip are included."""
    out: dict[tuple[str, str], int] = {}
    for key, dw in _dir_winners(rows).items():
        if len(set(dw.values())) < 2:
            continue  # consistent — not a flip
        out[key] = int(all(win == first for first, win in dw.items()))
    return out


def intrans_by_triad(rows, uids) -> dict[tuple[str, str, str], int]:
    """sorted uid triple -> 1 if the fully judged triad is intransitive."""
    return {trip: int(kind is not None)
            for trip, kind in intransitivity_by_triad(rows, uids).items()}


# --- Exact McNemar ---

def mcnemar(x: dict, y: dict) -> dict:
    """Exact McNemar on two {key -> 0/1} maps over their common keys: exact
    two-sided binomtest(min(b,c), b+c, 0.5) on the discordant pairs."""
    from scipy.stats import binomtest

    keys = x.keys() & y.keys()
    n = len(keys)
    n_x = sum(x[k] for k in keys)
    n_y = sum(y[k] for k in keys)
    b = sum(1 for k in keys if x[k] == 1 and y[k] == 0)
    c = sum(1 for k in keys if x[k] == 0 and y[k] == 1)
    disc = b + c
    p = binomtest(min(b, c), disc, 0.5).pvalue if disc else 1.0
    return {
        "n": n, "rate_x": n_x / n if n else float("nan"),
        "rate_y": n_y / n if n else float("nan"),
        "b": b, "c": c, "n_discordant": disc, "p": float(p),
    }


# --- Holm step-down across a family of p-values ---

def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


# --- Load + metrics once per corpus ---

def load_corpus(corpus: str):
    """Return judges (ordered as in T.ROWS), gold dict, and per-judge rows."""
    plan = [r for r in T.ROWS if r[0] == corpus]
    key = plan[0][1]
    gold = T.GOLD_LOADERS[key]()
    judges = [name for (_c, _k, _p, name) in plan]
    rows = {name: load_verdicts(path) for (_c, _k, path, name) in plan}
    return judges, gold, rows


def metrics_for_corpus(corpus: str) -> dict:
    """All per-judge paired views, BT skills and ρ for one corpus."""
    judges, gold, rows = load_corpus(corpus)
    ipp = {j: ipp_by_pair(rows[j]) for j in judges}
    flip = {j: flip_by_pair(rows[j]) for j in judges}
    # BT skill per judge on the uids that have gold.
    uid_sets = {
        j: {r["uid_a"] for r in rows[j]} | {r["uid_b"] for r in rows[j]}
        for j in judges
    }
    common_uids = sorted(set.intersection(*uid_sets.values()) & set(gold))
    g_arr = np.array([gold[u] for u in common_uids], float)
    skill = {j: bt_fit(common_uids, rows[j], alpha=T.ALPHA) for j in judges}
    intrans = {j: intrans_by_triad(rows[j], common_uids) for j in judges}
    rho_solo = {j: spearman(skill[j], g_arr) for j in judges}
    return {
        "corpus": corpus, "judges": judges, "gold": g_arr,
        "n_uids": len(common_uids), "ipp": ipp, "flip": flip,
        "intrans": intrans, "skill": skill, "rho_solo": rho_solo,
    }


def _in_scope(a: str, b: str, scope: str) -> bool:
    same = family(a) == family(b)
    return same if scope == "within" else (not same) if scope == "cross" else True


def within_pairs(judges) -> list[tuple[str, str]]:
    """Within-family (small, big) judge pairs in ROWS order (small first)."""
    return [(a, b) for a, b in itertools.combinations(judges, 2)
            if family(a) == family(b)]


def compute_pairs(M: dict, scope: str) -> list[dict]:
    """Per-pair paired tests (flip, IPP, intransitivity, Δρ) for one corpus,
    Holm-adjusted across the in-scope pairs per metric."""
    judges = M["judges"]
    pairs = [(a, b) for a, b in itertools.combinations(judges, 2)
             if _in_scope(a, b, scope)]
    recs = []
    for a, b in pairs:
        recs.append({
            "corpus": M["corpus"], "a": a, "b": b,
            "flip": mcnemar(M["flip"][a], M["flip"][b]),
            "ipp": mcnemar(M["ipp"][a], M["ipp"][b]),
            "intrans": mcnemar(M["intrans"][a], M["intrans"][b]),
            "rho": bootstrap_spearman_diff(M["skill"][a], M["skill"][b], M["gold"],
                                           B=B_BOOT, seed=SEED),
            "rho_a": M["rho_solo"][a], "rho_b": M["rho_solo"][b],
            "n_uids": M["n_uids"],
        })
    for metric, key in (("flip", "p"), ("ipp", "p"), ("intrans", "p"),
                        ("rho", "p_value")):
        ps = [r[metric][key] for r in recs]
        for r, padj in zip(recs, holm(ps)):
            r[metric]["p_holm"] = padj
    return recs


# --- Across-all-families pooled aggregate ---

def make_stratum(M: dict, a: str, b: str) -> dict:
    """One small→big step: the paired maps + BT skill needed to pool it."""
    return {
        "corpus": M["corpus"], "a": a, "b": b, "family": family(a),
        "flip_a": M["flip"][a], "flip_b": M["flip"][b],
        "ipp_a": M["ipp"][a], "ipp_b": M["ipp"][b],
        "intrans_a": M["intrans"][a], "intrans_b": M["intrans"][b],
        "skill_a": M["skill"][a], "skill_b": M["skill"][b], "gold": M["gold"],
    }


def pooled_mcnemar(strata: list[dict], ka: str, kb: str) -> dict:
    """Stratified exact McNemar: sum the discordant counts across the small→big
    strata, then an exact binomial test on the pool. Also returns the
    per-stratum (b, c)."""
    from scipy.stats import binomtest

    B = C = n = nx = ny = 0
    per = []
    for s in strata:
        x, y = s[ka], s[kb]
        keys = x.keys() & y.keys()
        bs = sum(1 for k in keys if x[k] == 1 and y[k] == 0)
        cs = sum(1 for k in keys if x[k] == 0 and y[k] == 1)
        n += len(keys)
        nx += sum(x[k] for k in keys)
        ny += sum(y[k] for k in keys)
        B += bs
        C += cs
        per.append((f"{s['corpus']}:{s['family']}", bs, cs))
    disc = B + C
    p = binomtest(min(B, C), disc, 0.5).pvalue if disc else 1.0
    return {
        "n": n, "rate_x": nx / n if n else float("nan"),
        "rate_y": ny / n if n else float("nan"),
        "b": B, "c": C, "n_discordant": disc, "p": float(p), "per_stratum": per,
    }


def pooled_bootstrap_rho(strata: list[dict], *, B: int = B_BOOT,
                         seed: int = SEED) -> dict:
    """Stratified paired item bootstrap of the mean Δρ (small − big) across the
    within-family strata."""
    from scipy.stats import spearmanr

    prep = []
    for s in strata:
        a = np.asarray(s["skill_a"], float)
        b = np.asarray(s["skill_b"], float)
        g = np.asarray(s["gold"], float)
        m = np.isfinite(a) & np.isfinite(b) & np.isfinite(g)
        prep.append((a[m], b[m], g[m]))
    diffs = [float(spearmanr(a, g).statistic - spearmanr(b, g).statistic)
             for a, b, g in prep]
    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    boots = np.empty(B, dtype=float)
    for i in range(B):
        acc = 0.0
        for a, b, g in prep:
            n = len(g)
            idx = rng.integers(0, n, n)
            gi = g[idx]
            acc += spearmanr(a[idx], gi).statistic - spearmanr(b[idx], gi).statistic
        boots[i] = acc / len(prep)
    return {
        "diff": mean_diff,
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "p_value": float(min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))),
        "per_stratum": list(zip((f"{s['corpus']}:{s['family']}" for s in strata),
                                diffs)),
        "n_strata": len(strata),
    }


def pooled_aggregate(strata: list[dict]) -> dict:
    return {
        "flip": pooled_mcnemar(strata, "flip_a", "flip_b"),
        "ipp": pooled_mcnemar(strata, "ipp_a", "ipp_b"),
        "intrans": pooled_mcnemar(strata, "intrans_a", "intrans_b"),
        "rho": pooled_bootstrap_rho(strata),
        "n_strata": len(strata),
        "steps": [f"{s['corpus']}:{s['a']}→{s['b']}" for s in strata],
    }


# --- Rendering ---

def _stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def text_report(all_recs: list[dict], scope: str, pooled: dict | None) -> str:
    out: list[str] = []
    for corpus in dict.fromkeys(r["corpus"] for r in all_recs):
        recs = [r for r in all_recs if r["corpus"] == corpus]
        out.append(f"\n=== {corpus} — {scope}-family judge pairs "
                   f"(n_uids={recs[0]['n_uids']}) ===")
        out.append("paired tests: flip, IPP & intrans = exact McNemar; "
                   "Δρ = paired item bootstrap (B=%d). p_holm within the %d pairs.\n"
                   % (B_BOOT, len(recs)))
        for r in recs:
            f, ip, it, rh = r["flip"], r["ipp"], r["intrans"], r["rho"]
            out.append(f"  {r['a']} vs {r['b']}")
            out.append(
                f"    flip    : {f['rate_x']:.3f} vs {f['rate_y']:.3f}  "
                f"Δ={f['rate_x'] - f['rate_y']:+.3f}  (b={f['b']}, c={f['c']}, "
                f"n_pairs={f['n']})  p={f['p']:.2e} {_stars(f['p'])}  "
                f"p_holm={f['p_holm']:.2e} {_stars(f['p_holm'])}"
            )
            out.append(
                f"    IPP     : {ip['rate_x']:.3f} vs {ip['rate_y']:.3f}  "
                f"Δ={ip['rate_x'] - ip['rate_y']:+.3f}  (b={ip['b']}, c={ip['c']}, "
                f"n_bothflip={ip['n']})  p={ip['p']:.2e} {_stars(ip['p'])}  "
                f"p_holm={ip['p_holm']:.2e} {_stars(ip['p_holm'])}"
            )
            out.append(
                f"    intrans : {it['rate_x']:.3f} vs {it['rate_y']:.3f}  "
                f"Δ={it['rate_x'] - it['rate_y']:+.3f}  (b={it['b']}, c={it['c']}, "
                f"n_triads={it['n']})  p={it['p']:.2e} {_stars(it['p'])}  "
                f"p_holm={it['p_holm']:.2e} {_stars(it['p_holm'])}"
            )
            out.append(
                f"    Δρ      : {rh['rho_a']:+.3f} vs {rh['rho_b']:+.3f}  "
                f"Δ={rh['diff']:+.3f}  95%CI[{rh['ci_lo']:+.3f}, {rh['ci_hi']:+.3f}]  "
                f"p={rh['p_value']:.2e} {_stars(rh['p_value'])}  "
                f"p_holm={rh['p_holm']:.2e} {_stars(rh['p_holm'])}"
            )
        out.append("")

    if pooled is not None:
        f, ip, it, rh = pooled["flip"], pooled["ipp"], pooled["intrans"], pooled["rho"]
        out.append(f"\n=== ACROSS ALL FAMILIES — pooled small→big, item-level "
                   f"({pooled['n_strata']} steps) ===")
        out.append("steps: " + ", ".join(pooled["steps"]))
        out.append("flip/IPP/intrans = stratified exact McNemar (discordants "
                   "pooled over steps); Δρ = stratified bootstrap of the mean "
                   "small→big ρ change.\n")
        out.append(
            f"  flip    : small {f['rate_x']:.3f} vs big {f['rate_y']:.3f}  "
            f"Δ={f['rate_x'] - f['rate_y']:+.3f}  (b={f['b']}, c={f['c']}, "
            f"n_pairs={f['n']})  p={f['p']:.2e} {_stars(f['p'])}"
        )
        out.append("            per-step (b,c): " + ", ".join(
            f"{lbl}={b}/{c}" for lbl, b, c in f["per_stratum"]))
        out.append(
            f"  IPP     : small {ip['rate_x']:.3f} vs big {ip['rate_y']:.3f}  "
            f"Δ={ip['rate_x'] - ip['rate_y']:+.3f}  (b={ip['b']}, c={ip['c']}, "
            f"n_bothflip={ip['n']})  p={ip['p']:.2e} {_stars(ip['p'])}"
        )
        out.append("            per-step (b,c): " + ", ".join(
            f"{lbl}={b}/{c}" for lbl, b, c in ip["per_stratum"]))
        out.append(
            f"  intrans : small {it['rate_x']:.3f} vs big {it['rate_y']:.3f}  "
            f"Δ={it['rate_x'] - it['rate_y']:+.3f}  (b={it['b']}, c={it['c']}, "
            f"n_triads={it['n']})  p={it['p']:.2e} {_stars(it['p'])}"
        )
        out.append("            per-step (b,c): " + ", ".join(
            f"{lbl}={b}/{c}" for lbl, b, c in it["per_stratum"]))
        out.append(
            f"  Δρ      : mean Δ={rh['diff']:+.3f}  "
            f"95%CI[{rh['ci_lo']:+.3f}, {rh['ci_hi']:+.3f}]  "
            f"p={rh['p_value']:.2e} {_stars(rh['p_value'])}"
        )
        out.append("            per-step Δρ: " + ", ".join(
            f"{lbl}={d:+.3f}" for lbl, d in rh["per_stratum"]))
        out.append("")
    return "\n".join(out)


def _md_p(p: float) -> str:
    """p-value + significance stars for a markdown cell, e.g. `<0.001 ***`."""
    txt = "<0.001" if p < 1e-3 else f"{p:.3f}"
    return f"{txt} {_stars(p)}"


def to_markdown(all_recs: list[dict], pooled: dict | None) -> str:
    """Markdown table, one row per judge pair: each metric's small−big Δ and
    its Holm-adjusted p. flip = flip rate (PC = 1 − flip). The pooled row
    carries the unadjusted stratified p."""
    head = ("| Corpus | Judge pair | Δflip | p | ΔIPP | p | Δintr. | p | Δρ | p |\n"
            "| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | --- |")
    lines = [head]
    last = None
    for r in all_recs:
        cell = f"**{r['corpus']}**" if r["corpus"] != last else ""
        last = r["corpus"]
        f, ip, it, rh = r["flip"], r["ipp"], r["intrans"], r["rho"]
        lines.append(
            f"| {cell} | {r['a']} vs {r['b']} "
            f"| {f['rate_x'] - f['rate_y']:+.3f} | {_md_p(f['p_holm'])} "
            f"| {ip['rate_x'] - ip['rate_y']:+.3f} | {_md_p(ip['p_holm'])} "
            f"| {it['rate_x'] - it['rate_y']:+.3f} | {_md_p(it['p_holm'])} "
            f"| {rh['diff']:+.3f} | {_md_p(rh['p_holm'])} |"
        )
    if pooled is not None:
        f, ip, it, rh = pooled["flip"], pooled["ipp"], pooled["intrans"], pooled["rho"]
        lines.append(
            f"| _all small→big (pooled, {pooled['n_strata']})_ | "
            f"| {f['rate_x'] - f['rate_y']:+.3f} | {_md_p(f['p'])} "
            f"| {ip['rate_x'] - ip['rate_y']:+.3f} | {_md_p(ip['p'])} "
            f"| {it['rate_x'] - it['rate_y']:+.3f} | {_md_p(it['p'])} "
            f"| {rh['diff']:+.3f} | {_md_p(rh['p_value'])} |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", type=Path, default=None,
                    help="write the markdown stats table here (also printed)")
    ap.add_argument("--scope", choices=("within", "cross", "both"),
                    default="within",
                    help="which judge pairs to compare (default: within-family)")
    args = ap.parse_args(argv)

    all_recs: list[dict] = []
    strata: list[dict] = []
    for corpus in dict.fromkeys(r[0] for r in T.ROWS):
        M = metrics_for_corpus(corpus)
        all_recs.extend(compute_pairs(M, args.scope))
        strata.extend(make_stratum(M, a, b) for a, b in within_pairs(M["judges"]))


    pooled = pooled_aggregate(strata) if args.scope in ("within", "both") else None

    print(text_report(all_recs, args.scope, pooled))
    md = to_markdown(all_recs, pooled)
    print(md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
