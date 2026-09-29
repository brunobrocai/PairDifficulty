"""Drawing the population and choosing which labels a judge gets wrong.

Both figure simulations start the same way: draw true abilities, then pick a
fixed number of comparisons to flip. Judges differ only in where those flips
land, which `pick_flips` controls.
"""
from __future__ import annotations

import numpy as np


def draw_beta(rng: np.random.Generator, population: str, n: int) -> np.ndarray:
    """True ability. Both options have variance 1, so |dbeta| has the same
    spread; only the shape differs (normal has a long tail, uniform does not)."""
    if population == "normal":
        return rng.standard_normal(n)
    return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), n)


def pick_flips(rng: np.random.Generator, delta_abs: np.ndarray, k: int,
               spread_share: float, k_close: float) -> np.ndarray:
    """Choose exactly k comparison indices to flip.

    A fraction `spread_share` of them is drawn uniformly (the nuisance errors
    that can hit any pair); the rest are drawn with weights 2**(-k_close*|dbeta|),
    so a pair's chance halves every 1/k_close of gap, which concentrates them
    on near-ties. The two draws are disjoint, so exactly k labels flip."""
    n = len(delta_abs)
    n_spread = int(round(spread_share * k))
    n_close = k - n_spread

    idx_spread = rng.choice(n, size=n_spread, replace=False)
    free = np.ones(n, dtype=bool)
    free[idx_spread] = False

    weights = 2.0 ** (-k_close * delta_abs)
    idx_close = draw_without_repeats(rng, weights, free, n_close)

    return np.concatenate([idx_spread, idx_close]).astype(np.int64)


def draw_without_repeats(rng: np.random.Generator, weights: np.ndarray,
                         free: np.ndarray, k: int) -> np.ndarray:
    """Draw k distinct indices, each with a chance proportional to its weight.

    Draws in batches with repeats allowed and keeps new indices in the order
    they first appear, which equals drawing one at a time without
    replacement."""
    left = free.copy()
    picked = []
    got = 0
    while got < k:
        pool = np.flatnonzero(left)
        w = weights[pool]
        # Draw about twice what is still missing, so most rounds finish it off.
        batch = rng.choice(len(pool), size=min(2 * (k - got), len(pool)),
                           replace=True, p=w / w.sum())
        _, first_seen = np.unique(batch, return_index=True)
        new = pool[batch[np.sort(first_seen)][:k - got]]
        picked.append(new)
        left[new] = False
        got += len(new)
    return np.concatenate(picked) if picked else np.empty(0, dtype=np.int64)
