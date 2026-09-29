"""Shared pieces of the paper's simulations.

The two figures in the paper are the same experiment with a different judge:

    human_calibrated_judges.py  Figure 1 -- the judge flips exactly k of the
                                human's labels, and only where they land changes
    blind_zone_judges.py        Figure 2 -- the judge coin-flips inside a blind
                                zone of near-ties instead of flipping labels

What they have in common lives here:

    bt          fitting Bradley-Terry and flipping verdicts
    population  drawing abilities, choosing which labels a judge gets wrong
    config      the settings the two figures must agree on
    plots       the printed table and the two figure layouts
    naming      settings -> file name
    cli         the command line and its guards
    links       logistic / probit / cauchy preference curves
    style       the colour scheme
"""
from __future__ import annotations

from . import bt, cli, config, links, naming, plots, population, style

__all__ = ["bt", "cli", "config", "links", "naming", "plots", "population",
           "style"]
