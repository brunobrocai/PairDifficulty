"""JSONL-backed cache + pinned-config manifest for resumable LLM runs.

A `CacheStore` is an append-only JSONL file. Each row is a single LLM
call result keyed by an opaque `call_id` string (pairwise:
"uid_a|uid_b"). Re-running a job:

    done = cache.existing_call_ids()
    for item in work:
        if cache_key(item) in done:
            continue
        ... call LLM ...
        cache.append({"call_id": ..., ...})

A `Manifest` is a single-shot JSON file that pins the run config. On
the first invocation it's written; on every subsequent invocation it's
hash-compared against the current config, and any drift in fatal
fields raises before a single LLM call is made.

`CachePaths` is the file-name convention shared across pilots —
cache_{judge}_{prompt_version}_{regime}[_{tag}].jsonl and friends —
exposed as a builder so pilots don't reinvent it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


class CacheStore:
    """Append-only JSONL cache, keyed by an opaque `call_id` string."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def existing_call_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        out: set[str] = set()
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.add(json.loads(line)["call_id"])
        return out

    def rows(self) -> Iterator[dict]:
        """Iterate over every row in insertion order. Duplicates kept."""
        if not self.path.exists():
            return
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def load_keyed(self) -> dict[str, dict]:
        """Return {call_id: row}. Last write wins on duplicates."""
        out: dict[str, dict] = {}
        for r in self.rows():
            out[r["call_id"]] = r
        return out

    def append(self, row: dict) -> None:
        if "call_id" not in row:
            raise ValueError("cache rows must carry a 'call_id' field")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self.existing_call_ids())

    def __repr__(self) -> str:
        n = len(self) if self.path.exists() else 0
        return f"CacheStore(path={self.path}, rows={n})"


DEFAULT_FATAL_FIELDS: tuple[str, ...] = (
    "judge",
    "method",
    "prompt_version",
    "prompt_hash",
    "regime",
    "sample_seed",
    "master_seed",
    "temperature",
    "reasoning_effort",
    "n",
)


class ManifestDriftError(RuntimeError):
    """Raised when a manifest's pinned config no longer matches the current run."""


class Manifest:
    """Pinned run config. Written on first run; hash-checked on rerun."""

    def __init__(
        self,
        path: Path,
        *,
        fatal_fields: tuple[str, ...] = DEFAULT_FATAL_FIELDS,
    ):
        self.path = Path(path)
        self.fatal_fields = fatal_fields

    def read(self) -> dict | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text())

    def migrate_legacy(self) -> Path | None:
        """Rename a pre-core legacy manifest (uppercase PROMPT_VERSION, no
        `method` field) to `.legacy.json` so the next write_or_check starts
        fresh. Returns the backup path, or None if no migration was needed.
        Cache rows are untouched (addressed by call_id); only the drift
        guard is reset.
        """
        prior = self.read()
        if prior is None:
            return None
        is_legacy = "PROMPT_VERSION" in prior and "method" not in prior
        if not is_legacy:
            return None
        backup = self.path.with_suffix(".legacy.json")
        self.path.rename(backup)
        return backup

    def write_or_check(self, current: dict) -> dict:
        """If no prior manifest, write `current` and return it. If a prior
        manifest exists, compare on `fatal_fields` and raise on drift.

        `extension_history`, if present, is treated as growth-only: the
        prior's history must be a prefix of the current's. Pair-schedule
        extensions append entries; growing this list is not drift."""
        prior = self.read()
        if prior is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(current, indent=2, ensure_ascii=False))
            return current
        drift = [k for k in self.fatal_fields if prior.get(k) != current.get(k)]
        if drift:
            raise ManifestDriftError(
                f"MANIFEST DRIFT — refusing to extend cache.\n"
                f"  manifest: {self.path}\n"
                f"  fields drifted: {drift}\n"
                f"  prior:   { {k: prior.get(k) for k in drift} }\n"
                f"  current: { {k: current.get(k) for k in drift} }\n"
                f"If something changed intentionally, bump prompt_version or "
                f"start a fresh cache (--tag, or new cache file)."
            )
        prior_hist = prior.get("extension_history") or []
        cur_hist = current.get("extension_history") or []
        if prior_hist and prior_hist != cur_hist[: len(prior_hist)]:
            raise ManifestDriftError(
                f"MANIFEST DRIFT — extension history diverged.\n"
                f"  manifest: {self.path}\n"
                f"  prior history has {len(prior_hist)} entries; current "
                f"does not contain them as a prefix. Extensions are "
                f"append-only — you cannot rewrite or reorder prior "
                f"extensions."
            )
        # Rewrite if any non-fatal field shifted (e.g. budget/history grew).
        if current != prior:
            self.path.write_text(json.dumps(current, indent=2, ensure_ascii=False))
            return current
        return prior


@dataclass(frozen=True)
class CachePaths:
    """File-name convention for a run's cache/manifest/extras.

    Layout (under `root`):
        cache_{stem}.jsonl
        manifest_{stem}.json
        schedule_{stem}.json   (pairwise only)
        ranking_{stem}.json    (pairwise only)

    where stem = "{judge_slug}_{prompt_version}[_{regime}][_{tag}]".
    The regime segment is dropped when `regime` is empty / None — useful
    for pilots with only one sample (e.g. HANNA) whose existing files
    use the no-regime convention.
    """

    root: Path
    judge: str
    prompt_version: str
    regime: str | None = None
    tag: str = ""

    @staticmethod
    def _judge_slug(model: str) -> str:
        return model.replace("/", "-").replace(".", "_")

    @property
    def stem(self) -> str:
        pieces = [self._judge_slug(self.judge), self.prompt_version]
        if self.regime:
            pieces.append(self.regime)
        if self.tag:
            pieces.append(self.tag)
        return "_".join(pieces)

    def cache(self) -> Path:
        return self.root / f"cache_{self.stem}.jsonl"

    def manifest(self) -> Path:
        return self.root / f"manifest_{self.stem}.json"

    def extra(self, kind: str) -> Path:
        """Pairwise auxiliaries: kind in {'schedule', 'ranking', ...}."""
        return self.root / f"{kind}_{self.stem}.json"
