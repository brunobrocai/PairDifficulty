"""The settings both figure simulations share.

Figure 1 (human_calibrated_judges.py) and Figure 2 (blind_zone_judges.py) are
the same experiment with a different judge.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import links as pl

N = 200                # items
N_PAIRS = 4000         # comparisons drawn per run
N_RUNS = 200
SEED = 0
DISAGREE = 0.40        # share of human labels each judge flips (identical for all)
K_CLOSE = 3.0          # a pair's weight halves every 1/3 of |dbeta|
SHARES = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]    # spread share q
BUCKETS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 10.0])
LABELS = [f"{lo:.2g}–{hi:.2g}" if hi < 9 else f"≥{lo:.2g}"
          for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:])]
NB = len(LABELS)
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
POPULATIONS = ("normal", "uniform")
# The human's preference. "bt" = the noisy coin, "deterministic" = the
# higher-ability text always wins.
HUMANS = ("bt", "deterministic")
LINKS = pl.LINKS       # shape of the human's preference curve
GOLDS = ("true", "human")   # what the ranking is scored against


# The settings of the paper's figures. Anything else is a variant.
PAPER_DEFAULTS = {
    "population": "normal",
    "human": "bt",
    "link": "logistic",
    "gold": "human",
    "disagree": DISAGREE,
    "k_close": K_CLOSE,
    "n_items": N,
    "n_pairs": N_PAIRS,
    "n_runs": N_RUNS,
    "seed": SEED,
    "shares": SHARES,
}


def _fmt(value) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(f"{float(v):g}" for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def non_default(cfg: dict) -> list[str]:
    """Which settings of this run differ from the paper's, as readable text.

    `cfg` is the config block of a results dict. Settings a script does not
    offer are skipped."""
    out = []
    for key, default in PAPER_DEFAULTS.items():
        if key not in cfg:
            continue
        value = cfg[key]
        if key == "shares":
            differs = [float(v) for v in value] != [float(v) for v in default]
        else:
            differs = value != default
        if differs:
            out.append(f"{key}={_fmt(value)} (paper: {_fmt(default)})")
    return out


def describe_run(cfg: dict, figure: str) -> str:
    """One banner saying whether this run is the paper figure or a variant."""
    diffs = non_default(cfg)
    if not diffs:
        return f"Settings: the paper default. This run reproduces {figure}."
    return (f"Settings: NOT the paper default, so this is a variant, not "
            f"{figure}.\n  changed: " + "\n           ".join(diffs))
