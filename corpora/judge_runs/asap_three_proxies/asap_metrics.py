"""Metrics for the ASAP three-proxy study. BT is fit with
`choix.ilsr_pairwise(alpha=0.01)`.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


# --- cache loader ---

def load_verdicts(cache_path: Path) -> list[dict]:
    """Load a JSONL pairwise cache, deduping by ``call_id`` (last write wins).

    Concurrent runs can leave duplicate rows. The AB and BA orderings of a
    pair have distinct call ids, so both survive.
    """
    by_id: dict[str, dict] = {}
    with Path(cache_path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_id[r["call_id"]] = r
    return list(by_id.values())


# --- correlations ---

def spearman(predictions: np.ndarray, gold: np.ndarray) -> float:
    """Spearman ρ, NaN-safe (drops pairs where either is NaN)."""
    from scipy.stats import spearmanr

    a = np.asarray(predictions, dtype=float)
    b = np.asarray(gold, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho)


def bootstrap_spearman_diff(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    gold: np.ndarray,
    *,
    B: int = 10_000,
    seed: int = 0,
) -> dict:
    """Paired item-bootstrap of the Spearman-ρ difference between two arms.

    Returns {rho_a, rho_b, diff, ci_lo, ci_hi, p_value, n} on the items
    finite in a, b and gold.
    """
    from scipy.stats import spearmanr

    a = np.asarray(preds_a, dtype=float)
    b = np.asarray(preds_b, dtype=float)
    g = np.asarray(gold, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(g)
    a, b, g = a[mask], b[mask], g[mask]
    n = int(len(g))
    nan = float("nan")
    base = {"rho_a": nan, "rho_b": nan, "diff": nan,
            "ci_lo": nan, "ci_hi": nan, "p_value": nan, "n": n}
    if n < 3:
        return base

    rho_a, _ = spearmanr(a, g)
    rho_b, _ = spearmanr(b, g)
    base["rho_a"], base["rho_b"] = float(rho_a), float(rho_b)
    base["diff"] = float(rho_a) - float(rho_b)

    rng = np.random.default_rng(seed)
    boots = np.empty(B, dtype=float)
    for i in range(B):
        idx = rng.integers(0, n, n)
        gi = g[idx]
        ra, _ = spearmanr(a[idx], gi)
        rb, _ = spearmanr(b[idx], gi)
        boots[i] = (ra - rb) if (np.isfinite(ra) and np.isfinite(rb)) else np.nan
    valid = boots[np.isfinite(boots)]
    if len(valid) < B // 2:
        return base
    base["ci_lo"] = float(np.percentile(valid, 2.5))
    base["ci_hi"] = float(np.percentile(valid, 97.5))
    frac_le = float(np.mean(valid <= 0.0))
    frac_ge = float(np.mean(valid >= 0.0))
    base["p_value"] = min(1.0, 2.0 * min(frac_le, frac_ge))
    return base


def bootstrap_rho_ci(
    preds: np.ndarray,
    gold: np.ndarray,
    *,
    B: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired item percentile bootstrap of ρ(preds, gold).

    Returns (rho, ci_lo, ci_hi).
    """
    from scipy.stats import spearmanr

    p = np.asarray(preds, dtype=float)
    g = np.asarray(gold, dtype=float)
    mask = np.isfinite(p) & np.isfinite(g)
    p, g = p[mask], g[mask]
    n = len(p)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    rho = spearman(p, g)
    rng = np.random.default_rng(seed)
    boots = np.empty(B, dtype=float)
    for i in range(B):
        idx = rng.integers(0, n, n)
        ri, _ = spearmanr(p[idx], g[idx])
        boots[i] = ri if np.isfinite(ri) else np.nan
    valid = boots[np.isfinite(boots)]
    if len(valid) < B // 2:
        return rho, float("nan"), float("nan")
    return rho, float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


def bootstrap_best_set(
    preds_by_arm: dict[str, np.ndarray],
    gold: np.ndarray,
    *,
    B: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """The 'best set' — arms not significantly worse than the top (MCB).

    Single-step max-statistic bootstrap with family-wise error control:
    the best set is the observed-best arm i* plus every arm whose gap-to-i*
    is within the (1-alpha) quantile of the resampled max centered gap.
    Returns {arms, n, best_arm, crit, rho, gap, p_adj, in_set}.
    """
    from scipy.stats import spearmanr

    arms = list(preds_by_arm)
    K = len(arms)
    g = np.asarray(gold, dtype=float)
    P = np.vstack([np.asarray(preds_by_arm[a], dtype=float) for a in arms]) \
        if K else np.empty((0, len(g)))
    mask = np.isfinite(g)
    for k in range(K):
        mask &= np.isfinite(P[k])
    P, g = P[:, mask], g[mask]
    n = int(len(g))
    nan = float("nan")
    base = {
        "arms": arms, "n": n, "best_arm": None, "crit": nan,
        "rho": {a: nan for a in arms}, "gap": {a: nan for a in arms},
        "p_adj": {a: nan for a in arms}, "in_set": {a: False for a in arms},
    }
    if K < 2 or n < 3:
        if K == 1 and n >= 3:
            a = arms[0]
            rho0, _ = spearmanr(P[0], g)
            base.update(best_arm=a, crit=0.0, rho={a: float(rho0)},
                        gap={a: 0.0}, p_adj={a: nan}, in_set={a: True})
        return base

    rho = np.array([spearmanr(P[k], g)[0] for k in range(K)], dtype=float)
    istar = int(np.nanargmax(rho))
    gap = rho[istar] - rho
    others = [k for k in range(K) if k != istar]

    rng = np.random.default_rng(seed)
    M = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, n)
        gi = g[idx]
        rb = np.array([spearmanr(P[k, idx], gi)[0] for k in range(K)], dtype=float)
        dev = (rb[istar] - rb[others]) - gap[others]
        dev = dev[np.isfinite(dev)]
        M[b] = float(np.max(dev)) if dev.size else np.nan

    valid = M[np.isfinite(M)]
    if len(valid) < B // 2:
        base.update(best_arm=arms[istar],
                    rho={arms[k]: float(rho[k]) for k in range(K)},
                    gap={arms[k]: float(gap[k]) for k in range(K)})
        base["in_set"][arms[istar]] = True
        return base

    crit = float(np.percentile(valid, 100.0 * (1.0 - alpha)))
    p_adj = {arms[k]: float(np.mean(valid >= gap[k])) for k in range(K)}
    p_adj[arms[istar]] = nan
    in_set = {arms[k]: bool(gap[k] <= crit) for k in range(K)}
    in_set[arms[istar]] = True
    return {
        "arms": arms, "n": n, "best_arm": arms[istar], "crit": crit,
        "rho": {arms[k]: float(rho[k]) for k in range(K)},
        "gap": {arms[k]: float(gap[k]) for k in range(K)},
        "p_adj": p_adj, "in_set": in_set,
    }


# --- Bradley-Terry fit ---

def bt_fit(
    uids: list[str],
    rows: Iterable[dict],
    *,
    alpha: float = 0.01,
    verdict_field: str = "verdict",
    uid_a_field: str = "uid_a",
    uid_b_field: str = "uid_b",
    verbose: bool = False,
) -> np.ndarray:
    """Fit Bradley-Terry skills from pairwise verdicts via choix.ilsr_pairwise.

    Returns centered log-skills aligned with `uids`. Raises on
    non-convergence.
    """
    import choix

    uid_to_idx = {u: i for i, u in enumerate(uids)}
    data: list[tuple[int, int]] = []
    for r in rows:
        v = r.get(verdict_field)
        a = r.get(uid_a_field)
        b = r.get(uid_b_field)
        if v is None or a is None or b is None:
            continue
        if a not in uid_to_idx or b not in uid_to_idx:
            continue
        winner, loser = (a, b) if v == "A" else (b, a)
        data.append((uid_to_idx[winner], uid_to_idx[loser]))
    if not data:
        return np.full(len(uids), float("nan"))

    n = len(uids)
    try:
        params = choix.ilsr_pairwise(n, data, alpha=alpha)
    except RuntimeError as e:
        raise RuntimeError(
            f"choix.ilsr_pairwise did not converge at alpha={alpha} "
            f"(n_uids={n}, n_comparisons={len(data)}): {e}. Most likely you "
            f"need more comparisons; only raise alpha if a stronger prior is "
            f"methodologically justified."
        )
    params = np.asarray(params, dtype=float)
    params = params - params.mean()
    if verbose:
        print(f"[BT fit] n_uids={n} n_comparisons={len(data)} alpha={alpha} "
              f"skill_range=[{params.min():+.3f}, {params.max():+.3f}]")
    return params


# --- consistency proxies ---

def _unordered_pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def swap_consistency(
    rows: Iterable[dict],
    *,
    uid_a_field: str = "uid_a",
    uid_b_field: str = "uid_b",
    verdict_field: str = "verdict",
) -> dict:
    """Order-swap self-consistency of a swap-augmented pairwise cache.

    Returns n_pairs_both_orders, flip_rate (fraction of both-order pairs whose
    directional majority winner flips), position_bias (P(verdict=="A")),
    n_decisive_calls, per_uid.
    """
    by_pair: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    n_decisive = 0
    n_a_pick = 0
    for r in rows:
        a = r.get(uid_a_field)
        b = r.get(uid_b_field)
        v = r.get(verdict_field)
        if a is None or b is None or v not in ("A", "B"):
            continue
        n_decisive += 1
        if v == "A":
            n_a_pick += 1
        winner = a if v == "A" else b
        by_pair[_unordered_pair_key(a, b)][a].append(winner)  # keyed by slot-A uid

    n_both = 0
    n_flip = 0
    per_uid: dict[str, dict] = defaultdict(
        lambda: {"n_swap_pairs": 0, "n_flips": 0}
    )
    for (u, v_), by_first in by_pair.items():
        if len(by_first) < 2:
            continue  # only one slot order present
        dir_winners = {
            first: Counter(ws).most_common(1)[0][0]
            for first, ws in by_first.items()
        }
        flipped = len(set(dir_winners.values())) > 1
        n_both += 1
        n_flip += int(flipped)
        per_uid[u]["n_swap_pairs"] += 1
        per_uid[v_]["n_swap_pairs"] += 1
        if flipped:
            per_uid[u]["n_flips"] += 1
            per_uid[v_]["n_flips"] += 1

    for u, d in per_uid.items():
        d["flip_rate"] = (
            d["n_flips"] / d["n_swap_pairs"] if d["n_swap_pairs"] else float("nan")
        )
    return {
        "n_pairs_both_orders": n_both,
        "flip_rate": n_flip / n_both if n_both else float("nan"),
        "position_bias": n_a_pick / n_decisive if n_decisive else float("nan"),
        "n_decisive_calls": n_decisive,
        "per_uid": dict(per_uid),
    }


def triad_intransitivity(
    rows: Iterable[dict],
    uids: list[str],
    skills: np.ndarray,
    *,
    uid_a_field: str = "uid_a",
    uid_b_field: str = "uid_b",
    verdict_field: str = "verdict",
) -> dict:
    """Soft intransitivity over every queried triad, with a BT baseline.

    For a triad with empirical win-fractions, `observed` = mean cyclic
    probability; `bt_expected` = same under the fitted transitive BT model;
    `excess = observed − bt_expected` is cycling not explained by a single
    latent scale plus sampling noise. Returns {n_triads, observed,
    bt_expected, excess}.
    """
    wins: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for r in rows:
        a = r.get(uid_a_field)
        b = r.get(uid_b_field)
        v = r.get(verdict_field)
        if a is None or b is None or v not in ("A", "B"):
            continue
        wins[_unordered_pair_key(a, b)][a if v == "A" else b] += 1

    wfrac: dict[tuple[str, str], float] = {}
    for (lo, hi), c in wins.items():
        tot = c.get(lo, 0) + c.get(hi, 0)
        if tot:
            wfrac[(lo, hi)] = c.get(lo, 0) / tot  # P(lo beats hi)

    idx = {u: i for i, u in enumerate(uids)}

    def w_emp(x: str, y: str) -> float | None:
        key = _unordered_pair_key(x, y)
        w = wfrac.get(key)
        if w is None:
            return None
        return w if key[0] == x else 1.0 - w

    def w_bt(x: str, y: str) -> float:
        return 1.0 / (1.0 + np.exp(-(skills[idx[x]] - skills[idx[y]])))

    n = len(uids)
    n_tri = 0
    obs_sum = 0.0
    exp_sum = 0.0
    for i in range(n):
        a = uids[i]
        for j in range(i + 1, n):
            b = uids[j]
            wab = w_emp(a, b)
            if wab is None:
                continue
            for k in range(j + 1, n):
                c = uids[k]
                wac = w_emp(a, c)
                if wac is None:
                    continue
                wbc = w_emp(b, c)
                if wbc is None:
                    continue
                n_tri += 1
                obs_sum += wab * wbc * (1 - wac) + wac * (1 - wbc) * (1 - wab)
                pab, pbc, pac = w_bt(a, b), w_bt(b, c), w_bt(a, c)
                exp_sum += pab * pbc * (1 - pac) + pac * (1 - pbc) * (1 - pab)
    if not n_tri:
        return {"n_triads": 0, "observed": float("nan"),
                "bt_expected": float("nan"), "excess": float("nan")}
    obs = obs_sum / n_tri
    exp = exp_sum / n_tri
    return {"n_triads": n_tri, "observed": obs, "bt_expected": exp,
            "excess": obs - exp}


def _weak_order_consistent(rab: int, rbc: int, rac: int) -> bool:
    """Is there a ranking-with-ties of (a,b,c) reproducing all three relations?

    Each relation is in {+1 (first beats second), -1 (second beats first),
    0 (tie)}. Brute-force over the 27 rank assignments (3 items, ≤3 levels —
    enough to express any weak order of three items)."""
    spec = ((0, 1, rab), (1, 2, rbc), (0, 2, rac))
    for ra in range(3):
        for rb in range(3):
            for rc in range(3):
                ranks = (ra, rb, rc)
                if all(((ranks[x] > ranks[y]) - (ranks[x] < ranks[y])) == r
                       for x, y, r in spec):
                    return True
    return False


# precomputed once: (rab, rbc, rac) -> consistent?  (27 keys)
_CONSISTENT = {
    (rab, rbc, rac): _weak_order_consistent(rab, rbc, rac)
    for rab in (-1, 0, 1) for rbc in (-1, 0, 1) for rac in (-1, 0, 1)
}


def _collapse_swap_relations(
    rows: Iterable[dict],
    *,
    uid_a_field: str = "uid_a",
    uid_b_field: str = "uid_b",
    verdict_field: str = "verdict",
) -> tuple[dict[tuple[str, str], int], int]:
    """Swap-collapse each unordered pair to one relation across its slot orders:
    +1 (lo≻hi), -1 (hi≻lo), or 0 (tie — the verdict flipped under swap, so the
    judge contradicts itself). Returns (rel keyed by (lo,hi), n_tie_pairs)."""
    wins: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for r in rows:
        a = r.get(uid_a_field)
        b = r.get(uid_b_field)
        v = r.get(verdict_field)
        if a is None or b is None or v not in ("A", "B"):
            continue
        wins[_unordered_pair_key(a, b)][a if v == "A" else b] += 1
    rel: dict[tuple[str, str], int] = {}  # (lo,hi) -> +1 lo≻hi, -1 hi≻lo, 0 tie
    n_tie_pairs = 0
    for (lo, hi), c in wins.items():
        nlo, nhi = c.get(lo, 0), c.get(hi, 0)
        if nlo and nhi:
            rel[(lo, hi)] = 0
            n_tie_pairs += 1
        elif nlo:
            rel[(lo, hi)] = 1
        elif nhi:
            rel[(lo, hi)] = -1
    return rel, n_tie_pairs


def _triad_kind(rab: int, rbc: int, rac: int) -> str | None:
    """The transitivity verdict for one triad's three swap-collapsed relations:
    None (consistent / weak-order embeddable) or the break kind, by tie count —
    `cyc` (0 ties, strict 3-cycle), `mix` (1 tie), `eq` (2 ties, a=b,b=c,a≠c).
    Does not depend on the order of (a,b,c)."""
    if _CONSISTENT[(rab, rbc, rac)]:
        return None
    n_ties = (rab == 0) + (rbc == 0) + (rac == 0)
    return "cyc" if n_ties == 0 else "eq" if n_ties == 2 else "mix"


def intransitivity_by_triad(
    rows: Iterable[dict],
    uids: list[str],
    *,
    uid_a_field: str = "uid_a",
    uid_b_field: str = "uid_b",
    verdict_field: str = "verdict",
) -> dict[tuple[str, str, str], str | None]:
    """Per-triad swap-collapsed transitivity verdict, keyed by the canonical
    (sorted) uid triple, over every triad whose three pairs were all judged.

    Value is None (consistent) or the break kind `cyc`/`mix`/`eq`. Kept per
    triad so two judges on the same schedule can be paired triad by triad."""
    rel, _ = _collapse_swap_relations(
        rows, uid_a_field=uid_a_field, uid_b_field=uid_b_field,
        verdict_field=verdict_field,
    )

    def rel_of(x: str, y: str) -> int | None:
        key = _unordered_pair_key(x, y)
        r = rel.get(key)
        if r is None:
            return None
        return r if key[0] == x else -r

    out: dict[tuple[str, str, str], str | None] = {}
    n = len(uids)
    for i in range(n):
        a = uids[i]
        for j in range(i + 1, n):
            b = uids[j]
            rab = rel_of(a, b)
            if rab is None:
                continue
            for k in range(j + 1, n):
                c = uids[k]
                rbc = rel_of(b, c)
                if rbc is None:
                    continue
                rac = rel_of(a, c)
                if rac is None:
                    continue
                out[tuple(sorted((a, b, c)))] = _triad_kind(rab, rbc, rac)
    return out


def triad_intransitivity_swap(
    rows: Iterable[dict],
    uids: list[str],
    *,
    uid_a_field: str = "uid_a",
    uid_b_field: str = "uid_b",
    verdict_field: str = "verdict",
) -> dict:
    """Hard triad intransitivity on the *swap-collapsed* relation.

    Each unordered pair is reduced across its slot orders to one relation:
    a **strict** winner if every decisive verdict names the same text, or a
    **tie** if the verdict flips under swap (the judge contradicts itself, so
    it is effectively indifferent). A triad {a,b,c} is **intransitive** iff its
    three relations admit no ranking-with-ties (weak order). That counts:
      - the strict 3-cycle (a≻b, b≻c, c≻a)            -> n_cycle
      - the indifference-chain break (a=b, b=c, a≠c)  -> n_equality_chain
      - mixed breaks (a=b, b≻c, but not a≻c)          -> n_mixed

    Returns {n_triads, n_intransitive, rate, n_cycle, n_equality_chain,
    n_mixed, n_pairs, n_tie_pairs, tie_pair_frac}. `n_triads` counts only
    triads whose three pairs were all judged (the schedule is sparse).
    """
    rel, n_tie_pairs = _collapse_swap_relations(
        rows, uid_a_field=uid_a_field, uid_b_field=uid_b_field,
        verdict_field=verdict_field,
    )
    by_triad = intransitivity_by_triad(
        rows, uids, uid_a_field=uid_a_field, uid_b_field=uid_b_field,
        verdict_field=verdict_field,
    )
    kinds = Counter(by_triad.values())
    n_tri = len(by_triad)
    n_cycle, n_eqchain, n_mixed = kinds["cyc"], kinds["eq"], kinds["mix"]
    n_intrans = n_cycle + n_eqchain + n_mixed
    n_pairs = len(rel)
    return {
        "n_triads": n_tri, "n_intransitive": n_intrans,
        "rate": n_intrans / n_tri if n_tri else float("nan"),
        "n_cycle": n_cycle, "n_equality_chain": n_eqchain, "n_mixed": n_mixed,
        "n_pairs": n_pairs, "n_tie_pairs": n_tie_pairs,
        "tie_pair_frac": n_tie_pairs / n_pairs if n_pairs else float("nan"),
    }


def verdict_accuracy(
    rows: Iterable[dict],
    uids: list[str],
    gold: np.ndarray,
    *,
    verdict_of: Callable[[dict], str | None],
    uid_a_field: str = "uid_a",
    uid_b_field: str = "uid_b",
) -> tuple[float, float, int]:
    """Per-pair agreement of a verdict with the gold-better text.

    Returns (acc_all, acc_clear, n); acc_clear restricts to pairs whose
    |gold_a − gold_b| exceeds the median margin. Pairs with equal gold are
    dropped entirely (no gold-better text).
    """
    idx = {u: i for i, u in enumerate(uids)}
    correct: list[float] = []
    margins: list[float] = []
    for r in rows:
        a = r.get(uid_a_field)
        b = r.get(uid_b_field)
        if a not in idx or b not in idx:
            continue
        v = verdict_of(r)
        if v not in ("A", "B"):
            continue
        ga, gb = gold[idx[a]], gold[idx[b]]
        if ga == gb:
            continue
        correct.append(1.0 if v == ("A" if ga > gb else "B") else 0.0)
        margins.append(abs(ga - gb))
    if not correct:
        return float("nan"), float("nan"), 0
    correct_a = np.asarray(correct)
    clear = np.asarray(margins) > np.median(margins)
    acc_clear = float(correct_a[clear].mean()) if clear.any() else float("nan")
    return float(correct_a.mean()), acc_clear, len(correct_a)


# --- metrics by gold gap ---

def swap_consistency_by_gap(
    rows: Iterable[dict],
    uids: list[str],
    gold: np.ndarray,
    *,
    uid_a_field: str = "uid_a",
    uid_b_field: str = "uid_b",
    verdict_field: str = "verdict",
) -> dict[int, dict]:
    """Order-swap flip rate bucketed by integer gold gap |gold_a − gold_b|.

    Same flip definition as `swap_consistency`, grouped by the gold-score gap
    of the pair. Gap 0 (gold ties) is kept.
    Returns {gap: {n_pairs_both, n_flips, flip_rate}}.
    """
    idx = {u: i for i, u in enumerate(uids)}
    by_pair: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        a = r.get(uid_a_field)
        b = r.get(uid_b_field)
        v = r.get(verdict_field)
        if a not in idx or b not in idx or v not in ("A", "B"):
            continue
        winner = a if v == "A" else b
        by_pair[_unordered_pair_key(a, b)][a].append(winner)

    out: dict[int, dict] = defaultdict(lambda: {"n_pairs_both": 0, "n_flips": 0})
    for (u, w), by_first in by_pair.items():
        if len(by_first) < 2:
            continue
        gap = int(abs(gold[idx[u]] - gold[idx[w]]))
        dir_winners = {
            first: Counter(ws).most_common(1)[0][0]
            for first, ws in by_first.items()
        }
        flipped = len(set(dir_winners.values())) > 1
        out[gap]["n_pairs_both"] += 1
        out[gap]["n_flips"] += int(flipped)
    for g, d in out.items():
        d["flip_rate"] = (
            d["n_flips"] / d["n_pairs_both"] if d["n_pairs_both"] else float("nan")
        )
    return dict(out)


def verdict_accuracy_by_gap(
    rows: Iterable[dict],
    uids: list[str],
    gold: np.ndarray,
    *,
    verdict_of: Callable[[dict], str | None] = lambda r: r.get("verdict"),
    uid_a_field: str = "uid_a",
    uid_b_field: str = "uid_b",
) -> dict[int, dict]:
    """Per-call verdict accuracy bucketed by integer gold gap (gap 0 dropped).

    For each decisive call on a pair with distinct gold, score whether the
    verdict picked the higher-gold text, and bucket by |gold_a − gold_b|.
    Returns {gap: {n_correct, n, acc}} with a Wilson CI added by the caller.
    """
    idx = {u: i for i, u in enumerate(uids)}
    out: dict[int, dict] = defaultdict(lambda: {"n_correct": 0, "n": 0})
    for r in rows:
        a = r.get(uid_a_field)
        b = r.get(uid_b_field)
        if a not in idx or b not in idx:
            continue
        v = verdict_of(r)
        if v not in ("A", "B"):
            continue
        ga, gb = gold[idx[a]], gold[idx[b]]
        if ga == gb:
            continue
        gap = int(abs(ga - gb))
        out[gap]["n"] += 1
        out[gap]["n_correct"] += int(v == ("A" if ga > gb else "B"))
    for g, d in out.items():
        d["acc"] = d["n_correct"] / d["n"] if d["n"] else float("nan")
    return dict(out)


def position_bias_by_gap(
    rows: Iterable[dict],
    uids: list[str],
    gold: np.ndarray,
    *,
    uid_a_field: str = "uid_a",
    uid_b_field: str = "uid_b",
    verdict_field: str = "verdict",
) -> dict[int, dict]:
    """Position bias P(verdict == first slot) bucketed by integer gold gap.
    Counted per call (both slot orders), not per pair. Gap 0 is kept. Returns
    {gap: {n_a, n, posbias}}."""
    idx = {u: i for i, u in enumerate(uids)}
    out: dict[int, dict] = defaultdict(lambda: {"n_a": 0, "n": 0})
    for r in rows:
        a = r.get(uid_a_field)
        b = r.get(uid_b_field)
        v = r.get(verdict_field)
        if a not in idx or b not in idx or v not in ("A", "B"):
            continue
        gap = int(abs(gold[idx[a]] - gold[idx[b]]))
        out[gap]["n"] += 1
        out[gap]["n_a"] += int(v == "A")
    for g, d in out.items():
        d["posbias"] = d["n_a"] / d["n"] if d["n"] else float("nan")
    return dict(out)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson 95% CI for a binomial proportion. Returns (mean, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)
