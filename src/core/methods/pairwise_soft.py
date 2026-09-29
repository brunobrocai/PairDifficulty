"""Pairwise via the soft-regular streaming scheduler.

Identical to `PairwiseRun` (prompt, parse, BT fit, ranking) except for the
schedule: `build_soft_regular_schedule` with a high `soft_x` (default 1024
= effectively round-robin / exactly d-regular at wave boundaries).

The win over `PairwiseRun`'s `--extend-by` is clean streaming: raising
`--budget` from K to K+M appends M disjoint pairs, and the union IS a fresh
budget=K+M run (the first K pairs are bit-equal), so the
"extended-graph-isn't-uniform" caveat disappears.

`--budget` is the primary knob and is grow-only on resume (shrinking
raises; start a fresh `--tag` cache for a smaller budget). `--soft-x` is
locked once a cache exists. Both are pinned in the manifest.
"""

from __future__ import annotations

import json
from typing import Any

from core.cache import DEFAULT_FATAL_FIELDS, Manifest
from core.data import DataItem
from core.methods.base import RunResult
from core.methods.pairwise import PairwiseRun
from core.pairing import build_soft_regular_schedule
from core.seeds import deterministic_shuffle


class PairwiseSoftRun(PairwiseRun):
    METHOD = "pairwise_soft"
    _INCREASE_BUDGET_HINT = "raise --budget"

    def __init__(
        self,
        *args,
        budget: int,
        soft_x: float = 1024.0,
        alpha: float = 0.01,
        pair_schedule_namespace: str | None = None,
        **kwargs,
    ):
        # Bypass PairwiseRun.__init__ (which requires degree); call BaseRun's
        # initializer directly via the grandparent.
        from core.methods.base import BaseRun
        BaseRun.__init__(self, *args, **kwargs)
        self.budget = budget
        self.soft_x = float(soft_x)
        self.alpha = alpha
        self.pair_schedule_namespace = pair_schedule_namespace
        self._pairs: list[tuple[str, str]] = []
        # Sentinels for inherited methods (after_execute, _fit_bt) that
        # reference self.degree / self.extend_by.
        n = len(self.sample.uids())
        self.degree = (2 * budget) // n if n else 0  # effective per-item d
        self.extend_by = 0
        self._extension_history: list[dict] = []  # unused
        # soft_x is drift-tracked; budget is grow-only (checked in _prepare_schedule).
        self.manifest = Manifest(
            self.paths.manifest(),
            fatal_fields=DEFAULT_FATAL_FIELDS + ("soft_x",),
        )

    def _base_namespace(self) -> str:
        return (
            self.pair_schedule_namespace
            or f"{self.sample.parent.name}|soft_schedule|{self.sample.regime}"
        )

    def _prepare_schedule(self) -> None:
        if self._pairs:
            return
        uids = self.sample.uids()
        ns = self._base_namespace()

        prior = self.manifest.read() or {}
        prior_budget = prior.get("budget")
        prior_soft_x = prior.get("soft_x")

        if prior_soft_x is not None and float(prior_soft_x) != self.soft_x:
            raise ValueError(
                f"soft_x changed from {prior_soft_x} (existing) to "
                f"{self.soft_x} (this run). soft_x is locked once a cache "
                f"exists; start a fresh cache (--tag) to vary it."
            )

        if prior_budget is not None and self.budget < prior_budget:
            raise ValueError(
                f"--budget shrunk from {prior_budget} (existing) to "
                f"{self.budget}. Budget is grow-only — start a fresh "
                f"cache (--tag) to use a smaller budget."
            )

        self._pairs = build_soft_regular_schedule(
            uids,
            budget=self.budget, x=self.soft_x,
            master_seed=self.cfg.master_seed, namespace=ns,
        )

        if prior_budget is not None and self.budget > prior_budget:
            added = self.budget - prior_budget
            print(
                f"[soft] budget grew from {prior_budget} to {self.budget} "
                f"(+{added} new pairs, prefix bit-equal to prior run)."
            )

    def extra_manifest_fields(self) -> dict[str, Any]:
        self._prepare_schedule()
        n = len(self.sample.uids())
        eff_degree = (2 * self.budget) / n if n else 0.0
        return {
            "budget": self.budget,
            "soft_x": self.soft_x,
            "effective_degree": eff_degree,
            "swap_augmented": True,
        }

    def shuffle_namespace(self) -> str:
        return f"pairwise_soft|{self.sample.regime}|pair_call_order"

    def execute(self, client) -> RunResult:
        self._prepare_schedule()
        if self.cfg.dry_run:
            items = self.build_work_items()
            items = deterministic_shuffle(
                items, self.cfg.master_seed, self.shuffle_namespace(),
            )
            if self.cfg.limit is not None:
                items = items[: self.cfg.limit]
            return self._dry_run(items)
        self._persist_schedule()
        # Skip PairwiseRun.execute; call the grandparent directly.
        from core.methods.base import BaseRun
        return BaseRun.execute(self, client)

    def _persist_schedule(self) -> None:
        path = self.paths.extra("schedule")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "n_pairs": len(self._pairs),
            "budget": self.budget,
            "soft_x": self.soft_x,
            "pairs": self._pairs,
        }, indent=2))
