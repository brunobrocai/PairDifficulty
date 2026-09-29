"""Fitting and simulating Bradley-Terry comparisons.

These four functions are the numerical core every simulation in this directory
shares: turn a set of pairwise verdicts into a wins matrix, fit Bradley-Terry
to it, and flip verdicts to build a judge.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def fit_bt_mm(wins: np.ndarray, max_iter: int = 1000, tol: float = 1e-8,
              ridge: float = 1e-3) -> np.ndarray:
    """Bradley-Terry MLE (MM algorithm) on a wins matrix (wins[i,j] = #(i beat j)).
    Returns mean-centered log-strengths."""
    n = wins.shape[0]
    pi = np.ones(n)
    W = wins.sum(axis=1).astype(float)
    n_ij = (wins + wins.T).astype(float)
    np.fill_diagonal(n_ij, 0.0)

    for _ in range(max_iter):
        denom_mat = n_ij / (pi[:, None] + pi[None, :])
        np.fill_diagonal(denom_mat, 0.0)
        denom = denom_mat.sum(axis=1) + ridge
        pi_new = (W + ridge) / denom
        pi_new = pi_new / pi_new.mean()
        if np.max(np.abs(pi_new - pi)) < tol:
            pi = pi_new
            break
        pi = pi_new

    theta = np.log(pi)
    return theta - theta.mean()


def close_flip_prob(delta_abs: np.ndarray, k: float, a: float) -> np.ndarray:
    """Concentrated near-tie errors. Capped at 0.5 (= random)."""
    return np.minimum(0.5, a * np.exp(-k * delta_abs))


def calibrate_close_a(delta_abs_all: np.ndarray, target: float, k: float) -> float:
    """Find scale `a` so expected flip rate over all pairs equals target."""
    def f(a: float) -> float:
        return close_flip_prob(delta_abs_all, k, a).mean() - target
    f_lo = f(0.0)
    if f_lo >= 0:
        return 0.0
    a_hi = 1.0
    while f(a_hi) < 0:
        a_hi *= 2
        if a_hi > 1e6:
            raise RuntimeError(f"Cannot reach target {target} with k={k}")
    return brentq(f, 0.0, a_hi, xtol=1e-8)


def simulate_judge(true_winner: np.ndarray, flip_probs: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    """observed_winner in {0,1} from true_winner with per-pair flip_probs."""
    flips = rng.random(len(flip_probs)) < flip_probs
    return np.where(flips, 1 - true_winner, true_winner)


def build_wins_matrix(n: int, pairs_i: np.ndarray, pairs_j: np.ndarray,
                      observed: np.ndarray) -> np.ndarray:
    """observed=1 → i beats j; observed=0 → j beats i."""
    W = np.zeros((n, n), dtype=np.int64)
    np.add.at(W, (pairs_i[observed == 1], pairs_j[observed == 1]), 1)
    np.add.at(W, (pairs_j[observed == 0], pairs_i[observed == 0]), 1)
    return W
