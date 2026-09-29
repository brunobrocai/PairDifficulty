"""Deterministic SHA256-based seeds and shuffles.

Every random decision in this codebase derives from a SHA256 of
`master_seed` plus a domain-namespace string. That means a rerun with
the same `master_seed` reproduces the same call order and the same
per-call API seeds — even if items were added or removed from the
sample (only items whose hash bucket they fall in change).
"""

from __future__ import annotations

import hashlib
import random
from typing import Iterable, TypeVar

T = TypeVar("T")


def _digest_int(s: str, nbits: int = 64) -> int:
    nbytes = nbits // 4  # hex chars
    return int(hashlib.sha256(s.encode()).hexdigest()[:nbytes], 16)


def seed_from_string(*parts: object) -> int:
    """Deterministic 64-bit int from a pipe-joined SHA256 of the parts."""
    return _digest_int("|".join(str(p) for p in parts), nbits=64)


def per_call_seed(master_seed: int, call_id: str) -> int:
    """32-bit seed for a single LLM call. Stable across reruns."""
    return _digest_int(f"{master_seed}|{call_id}", nbits=32)


def deterministic_shuffle(items: Iterable[T], *seed_parts: object) -> list[T]:
    """Shuffle a copy of `items` using a seed derived from `seed_parts`."""
    out = list(items)
    rng = random.Random(seed_from_string(*seed_parts))
    rng.shuffle(out)
    return out
