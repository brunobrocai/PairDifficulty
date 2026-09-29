"""ASAP 2.0 FACS dataset loader. Single quality dimension — the holistic FACS
essay score (integer 1-6). One population-quota sample (n=200) matched to the
natural score distribution. The judge sees the essay plus the shared FACS
assignment (in `extras`). Headline metric: Spearman ρ vs the human FACS score.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

from core.data import Data, DataItem, Sample

DEFAULT_ASAP2_DIR = Path.home() / "Data" / "LLM-as-judge" / "ASAP2"
CSV_NAME = "ASAP_2_Final_github_test.csv"
PROMPT_FILTER = "Facial action coding system"

# Natural FACS score distribution as a quota for n=200 (population regime).
# Twice the n=100 quota {1:1,2:23,3:41,4:26,5:7,6:2}.
POPULATION_QUOTA = {1: 2, 2: 46, 3: 82, 4: 52, 5: 14, 6: 4}
N = sum(POPULATION_QUOTA.values())  # 200


def resolve_asap2_dir(data_dir: Path | str | None = None) -> Path:
    """Resolve the ASAP2 data directory.

    Priority: explicit `data_dir` arg > ASAP2_DATA_DIR env var >
    ~/Data/LLM-as-judge/ASAP2.
    """
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    env = os.getenv("ASAP2_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_ASAP2_DIR


@lru_cache(maxsize=4)
def _facs_rows(data_dir: Path) -> list[dict]:
    csv = data_dir / CSV_NAME
    if not csv.exists():
        raise FileNotFoundError(
            f"ASAP2 dataset not found under {data_dir}. Expected "
            f"{CSV_NAME}. Set --data-dir or ASAP2_DATA_DIR to point at "
            f"the directory."
        )
    df = pd.read_csv(csv)
    df = df[df["prompt_name"] == PROMPT_FILTER].copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError(
            f"no rows with prompt_name=={PROMPT_FILTER!r} in {csv}"
        )
    # A few rows leave the shared `assignment` blank. Fill in the single
    # assignment everywhere; fail if there is more than one.
    canon = df["assignment"].dropna().astype(str).str.strip()
    canon = canon[canon != ""].unique()
    if len(canon) != 1:
        raise RuntimeError(
            f"expected exactly one FACS assignment, found {len(canon)}: "
            f"{[c[:60] for c in canon[:3]]}"
        )
    df["assignment"] = canon[0]
    return df[["essay_id", "score", "full_text", "assignment"]].to_dict(
        orient="records"
    )


class ASAP2FacsData(Data):
    """ASAP 2.0 FACS: source-based student essays, gold = holistic 1-6.

    Single regime (population-quota, n=200). `sample()` fills the natural
    distribution quota; `n` is ignored (the quota fixes the size).
    """

    def __init__(self, *, data_dir: Path | str | None = None):
        self.data_dir = resolve_asap2_dir(data_dir)
        super().__init__(name="asap2_facs", context={"dimension": "facs_holistic"})

    def load(self) -> Iterable[DataItem]:
        for r in _facs_rows(self.data_dir):
            yield DataItem(
                uid=str(r["essay_id"]),
                text=str(r["full_text"]).strip(),
                gold=float(r["score"]),
                extras={
                    "essay_id": str(r["essay_id"]),
                    "assignment": str(r["assignment"]).strip(),
                    "gold_score": int(r["score"]),
                },
            )

    def sample(self, *, n: int | None = None, seed: int = 0,
               regime: str = "") -> Sample:
        """Population-quota sample (n=200) at the natural FACS distribution.

        Deterministic sha256(seed, name, uid) ordering; walk it once,
        filling each score's quota in order. `n` is ignored (the quota
        fixes the size).
        """
        self._ensure_loaded()
        ordered = sorted(
            self._items,
            key=lambda it: hashlib.sha256(
                f"{seed}|{self.name}|{it.uid}".encode()
            ).hexdigest(),
        )
        counts = {s: 0 for s in POPULATION_QUOTA}
        picked: list[DataItem] = []
        for it in ordered:
            s = int(it.extras["gold_score"])
            if counts.get(s, 0) < POPULATION_QUOTA.get(s, 0):
                picked.append(it)
                counts[s] += 1
        short = {
            s: POPULATION_QUOTA[s] - counts[s]
            for s in POPULATION_QUOTA if counts[s] < POPULATION_QUOTA[s]
        }
        if short:
            raise RuntimeError(
                f"population quota underfilled: needed {POPULATION_QUOTA}, "
                f"got {counts}, short {short}"
            )
        return Sample(parent=self, items=picked, regime=regime, seed=seed)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="ASAP2 FACS loader smoke test")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="override ASAP2 data directory")
    args = p.parse_args()
    data = ASAP2FacsData(data_dir=args.data_dir)
    s = data.sample(seed=42)
    gold = s.gold_array()
    import numpy as np
    vals, cnts = np.unique(gold.astype(int), return_counts=True)
    print(f"asap2_facs: pool={len(data)}  sample n={len(s)}")
    print(f"  gold dist: {dict(zip(vals.tolist(), cnts.tolist()))}")
    print(f"  gold=[{gold.min():.0f}, {gold.max():.0f}]  mean={gold.mean():.2f}")
    a = s.items[0].extras["assignment"]
    print(f"  assignment: {len(a)} chars; head: {a[:120]!r}")
