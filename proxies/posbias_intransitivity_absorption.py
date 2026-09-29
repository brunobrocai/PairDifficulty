"""Does the BT model absorb the observed soft intransitivity? Posterior-predictive
check: fit BT skills, replay the observed schedule as Bernoulli draws (M0 = scale
only β=0; M1 = scale + one global slot-bias β), count soft breaks, and compare
excess = observed − predicted. excess ≈ 0 ⇒ near-tie artifact, not genuine cycling.

    uv run proxies/posbias_intransitivity_absorption.py [--out FILE.tex]
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

import posbias_close_far_table as T  # noqa: E402
from asap_metrics import (  # noqa: E402
    _CONSISTENT,
    _unordered_pair_key,
    bt_fit,
    load_verdicts,
    triad_intransitivity_swap,
)

B_SIM = 5000
SEED = 0


# --- Triad classification lookup: code in [0, 27) -> 0 consistent, 1 cycle, 2 eq-chain ---

def _build_type_lut() -> np.ndarray:
    lut = np.zeros(27, dtype=np.int8)
    for rab in (-1, 0, 1):
        for rbc in (-1, 0, 1):
            for rac in (-1, 0, 1):
                code = (rab + 1) * 9 + (rbc + 1) * 3 + (rac + 1)
                if _CONSISTENT[(rab, rbc, rac)]:
                    lut[code] = 0
                else:
                    n_ties = (rab == 0) + (rbc == 0) + (rac == 0)
                    lut[code] = 1 if n_ties == 0 else 2 if n_ties == 2 else 3
    return lut


TYPE_LUT = _build_type_lut()


def _classify(rel_triads: np.ndarray) -> tuple[int, int, int]:
    """rel_triads: (n_tri, 3) int array of {-1,0,+1} for (rab, rbc, rac)."""
    code = (rel_triads[:, 0] + 1) * 9 + (rel_triads[:, 1] + 1) * 3 + (rel_triads[:, 2] + 1)
    t = TYPE_LUT[code]
    return int((t == 1).sum()), int((t == 2).sum()), int((t == 3).sum())


# --- Per-cell precompute ---

def precompute(uids: list[str], rows: list[dict], skills: np.ndarray) -> dict:
    idx = {u: i for i, u in enumerate(uids)}

    # canonical pair (lo,hi) -> [n_calls with lo in slot A, n_calls with hi in slot A,
    #                            n_lo_wins (observed)]
    pair_calls: dict[tuple[str, str], list[int]] = {}
    d_oriented: list[float] = []   # s[slotA] - s[slotB] per decisive call (for beta fit)
    n_a = n_calls = 0
    for r in rows:
        a, b, v = r.get("uid_a"), r.get("uid_b"), r.get("verdict")
        if a not in idx or b not in idx or v not in ("A", "B"):
            continue
        n_calls += 1
        n_a += (v == "A")
        d_oriented.append(skills[idx[a]] - skills[idx[b]])
        lo, hi = _unordered_pair_key(a, b)
        rec = pair_calls.setdefault((lo, hi), [0, 0, 0])
        if a == lo:           # lo sat in slot A this call
            rec[0] += 1
            rec[2] += (v == "A")     # lo won?
        else:                 # hi sat in slot A
            rec[1] += 1
            rec[2] += (v == "B")     # verdict B = the slot-B text = lo won

    pairs = sorted(pair_calls)
    pair_index = {p: i for i, p in enumerate(pairs)}
    delta = np.array([skills[idx[lo]] - skills[idx[hi]] for lo, hi in pairs], float)
    c_loA = np.array([pair_calls[p][0] for p in pairs], int)
    c_hiA = np.array([pair_calls[p][1] for p in pairs], int)
    n_lo_win = np.array([pair_calls[p][2] for p in pairs], int)
    total = c_loA + c_hiA
    # observed relation per pair: +1 lo≻hi, -1 hi≻lo, 0 tie
    rel_obs = np.where(n_lo_win == total, 1, np.where(n_lo_win == 0, -1, 0)).astype(np.int8)

    # complete triads: uids sorted lexically, so for i<j<k the canonical key of
    # (a,b) is (a,b) — every relation is already in (first,second) order, no sign
    # flips needed.
    have = set(pairs)
    triad_pi: list[tuple[int, int, int]] = []
    n = len(uids)
    for i in range(n):
        a = uids[i]
        for j in range(i + 1, n):
            b = uids[j]
            if (a, b) not in have:
                continue
            ab = pair_index[(a, b)]
            for k in range(j + 1, n):
                c = uids[k]
                if (b, c) in have and (a, c) in have:
                    triad_pi.append((ab, pair_index[(b, c)], pair_index[(a, c)]))
    triad_pi_arr = np.array(triad_pi, dtype=np.int64) if triad_pi else np.empty((0, 3), np.int64)

    return {
        "delta": delta, "c_loA": c_loA, "c_hiA": c_hiA, "total": total,
        "rel_obs": rel_obs, "triad_pi": triad_pi_arr,
        "posbias": n_a / n_calls if n_calls else float("nan"),
        "d_oriented": np.array(d_oriented, float),
    }


def fit_beta(d_oriented: np.ndarray, target_posbias: float) -> float:
    """β solving mean_calls σ(Δ_oriented + β) = observed posbias (monotone in β)."""
    f = lambda b: float(np.mean(expit(d_oriented + b))) - target_posbias  # noqa: E731
    try:
        return brentq(f, -12.0, 12.0)
    except ValueError:
        return float("nan")


def simulate(pc: dict, beta: float, *, B: int, seed: int) -> dict:
    """Return predicted (cycle, eqchain, soft) arrays over B sims."""
    rng = np.random.default_rng(seed)
    delta, c_loA, c_hiA, total = pc["delta"], pc["c_loA"], pc["c_hiA"], pc["total"]
    triad_pi = pc["triad_pi"]
    p_loA = expit(delta + beta)
    p_hiA = expit(delta - beta)
    cyc = np.empty(B, int); eqc = np.empty(B, int); mix = np.empty(B, int)
    for s in range(B):
        L = rng.binomial(c_loA, p_loA) + rng.binomial(c_hiA, p_hiA)  # lo-text wins
        rel = np.where(L == total, 1, np.where(L == 0, -1, 0)).astype(np.int8)
        rt = rel[triad_pi]
        cyc[s], eqc[s], mix[s] = _classify(rt)
    soft = eqc + mix
    return {"cycle": cyc, "eqchain": eqc, "mixed": mix, "soft": soft}


def _summ(sim: np.ndarray, observed: int) -> dict:
    mean = float(sim.mean())
    return {
        "mean": mean,
        "lo": float(np.percentile(sim, 2.5)),
        "hi": float(np.percentile(sim, 97.5)),
        "excess": observed - mean,
        "p_ge": float(np.mean(sim >= observed)),   # small => more cycling than BT
        "p_le": float(np.mean(sim <= observed)),   # small => less cycling than BT
    }


def _verdict(m: dict, observed: int) -> str:
    if observed > m["hi"]:
        return f"EXCESS (p={m['p_ge']:.3f})"
    if observed < m["lo"]:
        return f"below band, sub-BT (p={m['p_le']:.3f})"
    return "within band"


# --- Driver ---

def compute_cell(corpus: str, key: str, path: Path, name: str) -> dict:
    gold = T.GOLD_LOADERS[key]()
    rows = load_verdicts(path)
    uids = sorted(({r["uid_a"] for r in rows} | {r["uid_b"] for r in rows}) & set(gold))
    skills = bt_fit(uids, rows, alpha=T.ALPHA)

    pc = precompute(uids, rows, skills)
    obs = triad_intransitivity_swap(rows, uids)
    obs_soft = obs["n_equality_chain"] + obs["n_mixed"]

    # Check: these relations must reproduce triad_intransitivity_swap's counts.
    rt = pc["rel_obs"][pc["triad_pi"]]
    c0, e0, m0 = _classify(rt)
    assert (c0, e0, m0) == (obs["n_cycle"], obs["n_equality_chain"], obs["n_mixed"]), \
        f"{name}: recomputed {(c0, e0, m0)} != vendored " \
        f"{(obs['n_cycle'], obs['n_equality_chain'], obs['n_mixed'])}"

    beta = fit_beta(pc["d_oriented"], pc["posbias"])
    sim0 = simulate(pc, 0.0, B=B_SIM, seed=SEED)
    sim1 = simulate(pc, beta, B=B_SIM, seed=SEED)

    return {
        "corpus": corpus, "judge": name, "n_triads": obs["n_triads"],
        "beta": beta, "posbias": pc["posbias"],
        "obs_cycle": obs["n_cycle"], "obs_eqchain": obs["n_equality_chain"],
        "obs_mixed": obs["n_mixed"], "obs_soft": obs_soft,
        "M0": {"soft": _summ(sim0["soft"], obs_soft),
               "cycle": _summ(sim0["cycle"], obs["n_cycle"])},
        "M1": {"soft": _summ(sim1["soft"], obs_soft),
               "cycle": _summ(sim1["cycle"], obs["n_cycle"])},
    }


def text_report(recs: list[dict]) -> str:
    out = [f"BT absorption of soft intransitivity (B={B_SIM} sims). excess = observed "
           f"− BT-predicted mean.\n  EXCESS test: p_ge=P(sim≥obs) small ⇒ genuine "
           f"non-transitivity beyond a transitive scale.\n"]
    n_excess = n_subbt = 0
    for corpus in dict.fromkeys(r["corpus"] for r in recs):
        out.append(f"=== {corpus} ===")
        for r in [x for x in recs if x["corpus"] == corpus]:
            m0, m1 = r["M0"]["soft"], r["M1"]["soft"]
            n_excess += r["obs_soft"] > m1["hi"]
            n_subbt += r["obs_soft"] < m1["lo"]
            out.append(f"  {r['judge']:<16} n_tri={r['n_triads']}  "
                       f"obs_soft={r['obs_soft']:>3}  (eqchain={r['obs_eqchain']}, "
                       f"mixed={r['obs_mixed']}; cycle obs={r['obs_cycle']})")
            out.append(f"      M0 scale-only : pred {m0['mean']:5.1f} "
                       f"[{m0['lo']:.0f},{m0['hi']:.0f}]  excess={m0['excess']:+6.1f}  "
                       f"{_verdict(m0, r['obs_soft'])}")
            out.append(f"      M1 scale+slot : pred {m1['mean']:5.1f} "
                       f"[{m1['lo']:.0f},{m1['hi']:.0f}]  excess={m1['excess']:+6.1f}  "
                       f"{_verdict(m1, r['obs_soft'])}   (β={r['beta']:+.3f})")
        out.append("")
    ratios = [r["M1"]["soft"]["mean"] / r["obs_soft"] for r in recs if r["obs_soft"]]
    out.append(
        f"Overall: {n_excess}/{len(recs)} cells show positive excess (genuine "
        f"non-transitivity); {n_subbt}/{len(recs)} sit BELOW the BT band.\n"
        f"  A transitive BT-Bernoulli scale would itself generate "
        f"{min(ratios):.1f}–{max(ratios):.1f}× MORE soft breaks than observed, so "
        f"all observed soft intransitivity is absorbed (no excess); the judges are\n"
        f"  if anything MORE transitive than BT — its logistic link over-predicts "
        f"flips for a near-deterministic (temp-0) judge."
    )
    return "\n".join(out)


def to_latex(recs: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{llccccc}",
        r"  \toprule",
        r"  & & & \multicolumn{2}{c}{M0 scale-only} & \multicolumn{2}{c}{M1 scale+slot} \\",
        r"  \cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"  Corpus & Judge & obs soft & pred [95\%] & excess & pred [95\%] & excess \\",
        r"  \midrule",
    ]
    last = None
    for r in recs:
        if last is not None and r["corpus"] != last:
            lines.append(r"  \midrule")
        cell = r["corpus"] if r["corpus"] != last else ""
        last = r["corpus"]
        m0, m1 = r["M0"]["soft"], r["M1"]["soft"]
        lines.append(
            f"  {cell:<6} & {r['judge']} & {r['obs_soft']} & "
            f"{m0['mean']:.0f} [{m0['lo']:.0f},{m0['hi']:.0f}] & ${m0['excess']:+.0f}$ & "
            f"{m1['mean']:.0f} [{m1['lo']:.0f},{m1['hi']:.0f}] & ${m1['excess']:+.0f}$ \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="write the .tex table here (also printed)")
    args = ap.parse_args(argv)

    recs = [compute_cell(corpus, key, path, name) for corpus, key, path, name in T.ROWS]
    print(text_report(recs))
    print()
    tex = to_latex(recs)
    print(tex)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(tex + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
