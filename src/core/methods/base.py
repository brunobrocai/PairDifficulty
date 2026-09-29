"""BaseRun: the shared iteration / cache / manifest / progress loop.

Subclasses (PairwiseRun, PairwiseSoftRun) implement four methods:

    build_work_items()          -> list[Any]
        Turn `self.sample` into the list of LLM calls to issue.
        Pairwise: one item per (uid_a, uid_b) ordered pair.

    cache_key(item) -> str
        Stable opaque string used as the call_id
        (pairwise: f"{a}|{b}").

    render_prompt(item) -> Prompt
        Pull `self.template` and substitute the right fields from
        item / sample / data.context.

    parse_response(parsed, item) -> dict
        Method-specific row fields (e.g. {'uid_a': a, 'uid_b': b,
        'verdict': v} for pairwise). Merged into the row written to
        the cache.

Optionally override:
    extra_manifest_fields()     -> dict   (e.g. {'rounds': 10} for pairwise)
    after_execute(...)          -> None   (e.g. fit BT + write ranking)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from tqdm import tqdm

from core.cache import CacheStore, CachePaths, Manifest
from core.data import Data, Sample
from core.judges import Judge, OpenAIJudge
from core.prompts import Prompt, PromptRegistry, PromptTemplate, REGISTRY
from core.seeds import deterministic_shuffle, per_call_seed


@dataclass
class RunConfig:
    """Per-run configuration, passed into a BaseRun subclass."""

    judge: str
    prompt_key: str
    prompt_version: str
    sample_seed: int = 0
    master_seed: int = 0
    temperature: float = 0.0
    reasoning_effort: str = "none"
    tag: str = ""
    limit: int | None = None
    dry_run: bool = False
    retry_failed: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    made: int
    skipped: int
    failed: int
    elapsed_sec: float
    dry_run: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class BaseRun(ABC):
    METHOD: ClassVar[str] = ""

    def __init__(
        self,
        *,
        data: Data,
        sample: Sample,
        cfg: RunConfig,
        cache_root: Path | str,
        registry: PromptRegistry | None = None,
    ):
        if not self.METHOD:
            raise TypeError(f"{type(self).__name__}.METHOD must be set")
        self.data = data
        self.sample = sample
        self.cfg = cfg
        self.registry = registry or REGISTRY
        self.template: PromptTemplate = self.registry.get(
            cfg.prompt_key, cfg.prompt_version
        )
        self.cache_root = Path(cache_root)
        self.paths = CachePaths(
            root=self.cache_root,
            judge=cfg.judge,
            prompt_version=cfg.prompt_version,
            regime=sample.regime,
            tag=cfg.tag,
        )
        self.cache = CacheStore(self.paths.cache())
        self.manifest = Manifest(self.paths.manifest())

    # ------- abstract: subclasses implement these -------

    @abstractmethod
    def build_work_items(self) -> list[Any]:
        ...

    @abstractmethod
    def cache_key(self, item: Any) -> str:
        ...

    @abstractmethod
    def render_prompt(self, item: Any) -> Prompt:
        ...

    @abstractmethod
    def parse_response(self, parsed: dict | None, item: Any) -> dict:
        ...

    # ------- overridable -------

    def extra_manifest_fields(self) -> dict[str, Any]:
        return {}

    def after_execute(self, result: RunResult) -> RunResult:
        return result

    def shuffle_namespace(self) -> str:
        return f"{self.METHOD}|{self.sample.regime}|run_order"

    def schema_name(self) -> str:
        return self.cfg.prompt_key.replace(".", "_")

    def is_success_row(self, row: dict) -> bool:
        """True if a cached row is a successful API result (vs a recorded
        failure with response_field=None). Override to use `--retry-failed`."""
        raise NotImplementedError(
            f"{type(self).__name__}.is_success_row must be implemented "
            f"to use --retry-failed."
        )

    # ------- orchestration -------

    def manifest_payload(self) -> dict[str, Any]:
        uids = self.sample.uids()
        payload = {
            "judge": self.cfg.judge,
            "method": self.METHOD,
            "prompt_version": self.cfg.prompt_version,
            "prompt_hash": self.template.content_hash,
            "regime": self.sample.regime,
            "sample_seed": self.cfg.sample_seed,
            "master_seed": self.cfg.master_seed,
            "temperature": self.cfg.temperature,
            "reasoning_effort": self.cfg.reasoning_effort,
            "n": len(self.sample),
            "uids_head": uids[:5],
            "n_uids_total": len(uids),
        }
        payload.update(self.extra_manifest_fields())
        return payload

    def execute(self, judge_or_client) -> RunResult:
        """Run the cache-filling loop. Accepts a Judge or a raw OpenAI
        client, which is wrapped in an OpenAIJudge using `cfg.judge`."""
        self.manifest.write_or_check(self.manifest_payload())

        items = self.build_work_items()
        items = deterministic_shuffle(
            items, self.cfg.master_seed, self.shuffle_namespace(),
        )
        if self.cfg.limit is not None:
            items = items[: self.cfg.limit]

        if self.cfg.dry_run:
            return self._dry_run(items)

        judge = self._coerce_judge(judge_or_client)

        if self.cfg.retry_failed:
            done = {
                cid for cid, row in self.cache.load_keyed().items()
                if self.is_success_row(row)
            }
        else:
            done = self.cache.existing_call_ids()
        n_made = n_skipped = n_failed = 0
        t0 = time.time()
        pbar = tqdm(
            items,
            desc=f"{self.METHOD}[{self.cfg.judge}|{self.sample.regime}]",
            unit="call",
            dynamic_ncols=True,
        )
        for item in pbar:
            cid = self.cache_key(item)
            if cid in done:
                n_skipped += 1
                pbar.set_postfix(made=n_made, cached=n_skipped, failed=n_failed)
                continue
            prompt = self.render_prompt(item)
            seed = per_call_seed(self.cfg.master_seed, cid)
            parsed, raw = judge.call(
                prompt,
                temperature=self.cfg.temperature,
                seed=seed,
                reasoning_effort=self.cfg.reasoning_effort,
                schema_name=self.schema_name(),
            )
            row = self._base_row(cid, seed, raw)
            row.update(self.parse_response(parsed, item))
            self.cache.append(row)
            done.add(cid)
            if parsed is None:
                n_failed += 1
            else:
                n_made += 1
            pbar.set_postfix(made=n_made, cached=n_skipped, failed=n_failed)
        pbar.close()

        result = RunResult(
            made=n_made, skipped=n_skipped, failed=n_failed,
            elapsed_sec=time.time() - t0,
        )
        print(
            f"done: made={result.made} cached={result.skipped} "
            f"failed={result.failed} elapsed={result.elapsed_sec:.1f}s"
        )
        return self.after_execute(result)

    def _coerce_judge(self, judge_or_client) -> Judge:
        """Accept either a Judge or a raw OpenAI client."""
        if isinstance(judge_or_client, Judge):
            return judge_or_client
        return OpenAIJudge(self.cfg.judge, client=judge_or_client)

    def _base_row(self, cid: str, seed: int, raw: dict) -> dict[str, Any]:
        sysfp = raw.get("system_fingerprint") if isinstance(raw, dict) else None
        return {
            "call_id": cid,
            "judge": self.cfg.judge,
            "method": self.METHOD,
            "regime": self.sample.regime,
            "prompt_version": self.cfg.prompt_version,
            "temperature": self.cfg.temperature,
            "seed": seed,
            "system_fingerprint": sysfp,
        }

    def _dry_run(self, items: list[Any]) -> RunResult:
        if not items:
            print("DRY RUN — no items.")
            return RunResult(made=0, skipped=0, failed=0, elapsed_sec=0.0, dry_run=True)
        item = items[0]
        prompt = self.render_prompt(item)
        cid = self.cache_key(item)
        seed = per_call_seed(self.cfg.master_seed, cid)
        print("=" * 70)
        print(
            f"DRY RUN — {self.METHOD}  judge={self.cfg.judge}  "
            f"regime={self.sample.regime}  call_id={cid}  seed={seed}"
        )
        print("=" * 70)
        print("--- SYSTEM ---")
        print(prompt.sysprompt)
        print("--- USER ---")
        text = prompt.userprompt
        cap = 2000
        if len(text) > cap:
            print(text[:cap])
            print(f"... [truncated, full length={len(text)} chars]")
        else:
            print(text)
        return RunResult(
            made=0, skipped=0, failed=0, elapsed_sec=0.0,
            dry_run=True, extra={"n_items": len(items)},
        )
