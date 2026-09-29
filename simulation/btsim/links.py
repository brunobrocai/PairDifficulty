"""How a human's preference probability depends on the ability gap.

The paper assumes Bradley-Terry: P(i beats j) = sigmoid(beta_i - beta_j).
The alternatives here test other curve shapes:

    logistic  Bradley-Terry, the paper's default.
    probit    Thurstone-Mosteller Case V: Gaussian noise, lighter tails.
    cauchy    Cauchy noise, heavy tails: even large gaps are sometimes called
              the wrong way.

All three give a 75% win probability at a gap of ln(3), so they differ only
in the tails.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

LINKS = ("logistic", "probit", "cauchy")

# Gap at which the logistic link reaches a 75% win probability.
REF_GAP = float(np.log(3.0))
# Scales that put the other two links through the same point.
PROBIT_SCALE = REF_GAP / float(norm.ppf(0.75))     # ~1.629
CAUCHY_SCALE = REF_GAP                             # tan(pi/4) = 1


def win_prob(delta: np.ndarray, link: str = "logistic") -> np.ndarray:
    """P(the human picks the higher-ability item), given the ability gap."""
    if link == "logistic":
        return 1.0 / (1.0 + np.exp(-delta))
    if link == "probit":
        return norm.cdf(delta / PROBIT_SCALE)
    if link == "cauchy":
        return 0.5 + np.arctan(delta / CAUCHY_SCALE) / np.pi
    raise ValueError(f"unknown link {link!r}, expected one of {LINKS}")
