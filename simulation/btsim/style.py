"""One colour scheme for every figure in simulation/.

Colour-blind safe. SPECIALIST (q = 0) is blue; the spread judges (q > 0) use
a warm ramp, light for small q and dark for large q. HUMAN and GRID are greys
for reference lines. EXTRA holds colours for plots with unordered curves.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# The specialist / q = 0 judge.
SPECIALIST = "#0072B2"

# Neutral reference lines (the human, gridlines, annotation text).
HUMAN = "#6E6E6E"

# Unordered categorical curves: blue, vermillion, bluish green.
EXTRA = ["#0072B2", "#D55E00", "#009E73"]

# The warm ramp for the spread judges, as a slice of matplotlib's Oranges.

_RAMP = plt.cm.Oranges
_RAMP_LO, _RAMP_HI = 0.60, 1.00


def spread_color(i: int, n: int):
    """Colour for spread judge `i` of `n`, light (small floor) to dark (large)."""
    if n <= 1:
        return _RAMP(_RAMP_HI)
    return _RAMP(_RAMP_LO + (_RAMP_HI - _RAMP_LO) * i / (n - 1))
