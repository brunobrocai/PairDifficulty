"""Pairwise: one LLM call per (uid_a, uid_b) ordered pair.

Pairs are scheduled via a random d-regular graph (every uid gets
exactly `degree` comparisons). Every unordered pair is queried in
both orders ("swap augmentation") to net out position bias. After the
run, a Bradley-Terry skill is fit and a ranking JSON is written
alongside the cache.

Cache key: f"{uid_a}|{uid_b}".
Renders the prompt with `text_a, text_b, uid_a, uid_b`, plus
`a_<extra>` and `b_<extra>` for per-item extras, plus shared
extras under their natural name when they match across both items.
"""

from __future__ import annotations

import json
from typing import Any

from core.cache import DEFAULT_FATAL_FIELDS, Manifest
from core.data import DataItem
from core.methods.base import BaseRun, RunResult
from core.pairing import build_extension_schedule, build_schedule
from core.prompts import Prompt
from core.seeds import deterministic_shuffle


class PairwiseRun(BaseRun):
    METHOD = "pairwise"

    # Overridden by PairwiseSoftRun so the BT-fit "didn't converge"
    # error message points at the right CLI knob.
    _INCREASE_BUDGET_HINT: str = "raise --degree or --n"

    def __init__(
        self,
        *args,
        degree: int = 10,
        extend_by: int = 0,
        alpha: float = 0.01,
        pair_schedule_namespace: str | None = None,
        **kwargs,
    ):
        """`degree` is the base comparison count per item (fixed once a
        cache exists). `extend_by`, if > 0, adds that many new comparisons
        per item, disjoint from every prior schedule."""
        super().__init__(*args, **kwargs)
        self.degree = degree
        self.extend_by = extend_by
        self.alpha = alpha
        self.pair_schedule_namespace = pair_schedule_namespace
        self._pairs: list[tuple[str, str]] = []
        self._extension_history: list[dict] = []
        # Lock `degree` (base) as a drift-tracked field. Extensions grow
        # extension_history (allowed); shrinking or replacing the base
        # is a config drift, not a resume.
        self.manifest = Manifest(
            self.paths.manifest(),
            fatal_fields=DEFAULT_FATAL_FIELDS + ("degree",),
        )
        # Schedule is materialised lazily in execute() so __init__ has
        # no side effects on the manifest/schedule files.

    def _base_namespace(self) -> str:
        return (
            self.pair_schedule_namespace
            or f"{self.sample.parent.name}|pair_schedule|{self.sample.regime}"
        )

    def _prepare_schedule(self) -> None:
        """Materialise self._pairs and self._extension_history from
        (manifest + this invocation's --extend-by). Idempotent."""
        if self._pairs:
            return
        uids = self.sample.uids()
        ns = self._base_namespace()

        prior = self.manifest.read() or {}
        prior_history = prior.get("extension_history") or []
        prior_base_degree = prior.get("degree")

        if prior_base_degree is not None and prior_base_degree != self.degree:
            raise ValueError(
                f"--degree changed from {prior_base_degree} (existing) "
                f"to {self.degree} (this run). Base degree is locked once "
                f"a cache exists; add comparisons via --extend-by instead, "
                f"or start a fresh cache (--tag)."
            )

        base_pairs = build_schedule(
            uids, degree=self.degree, master_seed=self.cfg.master_seed,
            namespace=ns,
        )
        all_pairs: list[tuple[str, str]] = list(base_pairs)
        for ext in prior_history:
            ext_pairs = build_extension_schedule(
                all_pairs, uids,
                degree=ext["added_degree"],
                master_seed=self.cfg.master_seed,
                namespace=ext["namespace"],
            )
            all_pairs.extend(ext_pairs)

        new_history = list(prior_history)
        if self.extend_by > 0:
            if prior_base_degree is None:
                raise ValueError(
                    "--extend-by requires an existing cache. Run "
                    "--degree N first (a fresh base run) before extending."
                )
            ext_index = len(prior_history) + 1
            ext_ns = f"{ns}|ext{ext_index}"
            ext_pairs = build_extension_schedule(
                all_pairs, uids,
                degree=self.extend_by,
                master_seed=self.cfg.master_seed,
                namespace=ext_ns,
            )
            new_history.append({
                "ext_index": ext_index,
                "added_degree": self.extend_by,
                "namespace": ext_ns,
                "n_pairs_added": len(ext_pairs),
            })
            all_pairs.extend(ext_pairs)
            print(
                f"[extension] ext{ext_index}: +{self.extend_by} comparisons "
                f"per item ({len(ext_pairs)} new pairs disjoint from "
                f"{len(all_pairs) - len(ext_pairs)} prior)."
            )

        self._pairs = all_pairs
        self._extension_history = new_history

    def build_work_items(self) -> list[tuple[DataItem, DataItem]]:
        self._prepare_schedule()
        ordered: list[tuple[DataItem, DataItem]] = []
        for a_uid, b_uid in self._pairs:
            a = self.sample[a_uid]
            b = self.sample[b_uid]
            ordered.append((a, b))
            ordered.append((b, a))
        return ordered

    def cache_key(self, item: tuple[DataItem, DataItem]) -> str:
        a, b = item
        return f"{a.uid}|{b.uid}"

    def render_prompt(self, item: tuple[DataItem, DataItem]) -> Prompt:
        a, b = item
        fields: dict[str, Any] = {
            **self.data.context,
            "text_a": a.text, "uid_a": a.uid,
            "text_b": b.text, "uid_b": b.uid,
        }
        for k, v in a.extras.items():
            fields[f"a_{k}"] = v
        for k, v in b.extras.items():
            fields[f"b_{k}"] = v
        # Expose shared extras under the natural name (e.g. `assignment`
        # in datasets where every item has the same writing prompt).
        for k in set(a.extras) & set(b.extras):
            if a.extras[k] == b.extras[k]:
                fields.setdefault(k, a.extras[k])
        return self.template.render(**fields)

    def parse_response(
        self, parsed: dict | None, item: tuple[DataItem, DataItem],
    ) -> dict[str, Any]:
        a, b = item
        field = self.cfg.extra.get("response_field", "verdict")
        value = parsed.get(field) if parsed else None
        # Normalize verdict to "A" / "B" (or None).
        if isinstance(value, str):
            v = value.strip().upper()
            value = v if v in ("A", "B") else None
        return {"uid_a": a.uid, "uid_b": b.uid, field: value}

    def is_success_row(self, row: dict) -> bool:
        field = self.cfg.extra.get("response_field", "verdict")
        return row.get(field) in ("A", "B")

    def extra_manifest_fields(self) -> dict[str, Any]:
        self._prepare_schedule()
        out: dict[str, Any] = {"degree": self.degree, "swap_augmented": True}
        if self._extension_history:
            out["total_degree"] = self.degree + sum(
                e["added_degree"] for e in self._extension_history
            )
            out["extension_history"] = self._extension_history
        return out

    def shuffle_namespace(self) -> str:
        return f"pairwise|{self.sample.regime}|pair_call_order"

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
        return super().execute(client)

    def _persist_schedule(self) -> None:
        path = self.paths.extra("schedule")
        path.parent.mkdir(parents=True, exist_ok=True)
        total_degree = self.degree + sum(
            e["added_degree"] for e in self._extension_history
        )
        path.write_text(json.dumps({
            "n_pairs": len(self._pairs),
            "base_degree": self.degree,
            "total_degree": total_degree,
            "extension_history": self._extension_history,
            "pairs": self._pairs,
        }, indent=2))

    def after_execute(self, result: RunResult) -> RunResult:
        if result.dry_run:
            return result
        stats = self._fit_bt()
        if stats is None:
            print("BT fit skipped (no valid verdicts).")
            return result
        ranking = sorted(stats.items(), key=lambda kv: (-kv[1]["bt_skill"], kv[0]))
        ranking_uids = [u for u, _ in ranking]
        n_swap_disagree = sum(s["n_swap_disagree"] for s in stats.values()) // 2
        path = self.paths.extra("ranking")
        path.write_text(json.dumps({
            "ranking": ranking_uids,
            "stats": stats,
            "judge": self.cfg.judge,
            "regime": self.sample.regime,
            "degree": self.degree,
            "alpha": self.alpha,
            "aggregator": "bradley_terry_ilsr",
        }, indent=2))
        skills = [s["bt_skill"] for s in stats.values()]
        print(
            f"  swap-order disagreements: {n_swap_disagree} / {len(self._pairs)} "
            f"pairs ({n_swap_disagree / max(len(self._pairs), 1):.1%})"
        )
        print(f"  BT skill range: [{min(skills):+.3f}, {max(skills):+.3f}]")
        print(f"  ranking written -> {path}")
        result.extra["ranking_path"] = str(path)
        result.extra["n_swap_disagree"] = n_swap_disagree
        return result

    def _fit_bt(self) -> dict[str, dict] | None:
        """Fit a Bradley-Terry model to the swap-augmented verdicts."""
        import choix

        verdicts = {
            r["call_id"]: r.get(self.cfg.extra.get("response_field", "verdict"))
            for r in self.cache.rows()
        }
        uids = self.sample.uids()
        uid_to_idx = {u: i for i, u in enumerate(uids)}
        n = len(uids)

        data: list[tuple[int, int]] = []
        raw_stats: dict[str, dict] = {
            u: {"wins": 0, "trials": 0, "n_swap_disagree": 0} for u in uids
        }
        for a, b in self._pairs:
            v_ab = verdicts.get(f"{a}|{b}")
            v_ba = verdicts.get(f"{b}|{a}")
            if v_ab is None and v_ba is None:
                continue
            for v, first, second in ((v_ab, a, b), (v_ba, b, a)):
                if v is None:
                    continue
                winner = first if v == "A" else second
                loser = second if winner == first else first
                data.append((uid_to_idx[winner], uid_to_idx[loser]))
                raw_stats[winner]["wins"] += 1
                raw_stats[winner]["trials"] += 1
                raw_stats[loser]["trials"] += 1
            if v_ab is not None and v_ba is not None:
                picked_ab = a if v_ab == "A" else b
                picked_ba = b if v_ba == "A" else a
                if picked_ab != picked_ba:
                    raw_stats[a]["n_swap_disagree"] += 1
                    raw_stats[b]["n_swap_disagree"] += 1
        if not data:
            return None

        try:
            params = choix.ilsr_pairwise(n, data, alpha=self.alpha)
        except RuntimeError as e:
            raise RuntimeError(
                f"choix.ilsr_pairwise did not converge at alpha={self.alpha} "
                f"(n_uids={n}, n_comparisons={len(data)}): {e}. "
                f"Most likely you need more comparisons "
                f"({self._INCREASE_BUDGET_HINT}); only raise --alpha if "
                f"you've decided a stronger prior is methodologically "
                f"justified."
            )
        params = params - params.mean()
        print(
            f"[BT fit] n_uids={n} n_comparisons={len(data)} "
            f"alpha={self.alpha} method=ilsr "
            f"skill_range=[{params.min():+.3f}, {params.max():+.3f}]"
        )

        out: dict[str, dict] = {}
        for u in uids:
            rs = raw_stats[u]
            idx = uid_to_idx[u]
            out[u] = {
                "bt_skill": float(params[idx]),
                "wins": rs["wins"],
                "trials": rs["trials"],
                "win_rate": (
                    rs["wins"] / rs["trials"] if rs["trials"] else float("nan")
                ),
                "n_swap_disagree": rs["n_swap_disagree"],
            }
        return out
