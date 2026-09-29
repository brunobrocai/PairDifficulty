"""Locked seed constants — the single source of truth for any RNG seed
that ends up in a reported number. Mirrors `preregistration/seeds.md`.

Rule: one seed required → 42. n seeds required → 0, 1, ..., n-1.

Anything that touches a reported number must import its default from
here, not hardcode a literal. Debug / smoke runs may override via CLI;
the preregistration covers reported numbers only (see seeds.md).
"""

from __future__ import annotations

SINGLE_SEED: int = 42

SAMPLE_SEED: int = SINGLE_SEED
MASTER_SEED: int = SINGLE_SEED
BOOTSTRAP_SEED: int = SINGLE_SEED


def seeds_for(n: int) -> list[int]:
    """Return the locked contiguous seed list for the n-seed case.

    `seeds_for(1) == [0]`, `seeds_for(5) == [0, 1, 2, 3, 4]`. Use this
    whenever you need multiple seeds (e.g. multi-seed ablation, seed
    sensitivity). For single-seed slots use `SINGLE_SEED` (= 42).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return list(range(n))
