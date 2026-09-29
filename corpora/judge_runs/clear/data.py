"""CLEAR (CommonLit Ease of Readability) loader. Single quality construct
(readability ease); gold = `BT_easiness` (continuous). One population sample
(n=200) drawn as an equal-probability random subsample of the natural easiness
distribution. The judge sees the excerpt only. Headline metric: Spearman ρ vs gold.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

from core.data import Data, DataItem, Sample

DEFAULT_CLEAR_DIR = Path.home() / "Data" / "LLM-as-judge" / "CLEAR"
XLSX_NAME = "CLEAR_corpus_final.xlsx"

# Sample size used in the paper (run.py rejects --n).
N = 200


def resolve_clear_dir(data_dir: Path | str | None = None) -> Path:
    """Resolve the CLEAR data directory.

    Priority: explicit `data_dir` arg > CLEAR_DATA_DIR env var >
    ~/Data/LLM-as-judge/CLEAR.
    """
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    env = os.getenv("CLEAR_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_CLEAR_DIR


@lru_cache(maxsize=4)
def _clear_rows(data_dir: Path) -> list[dict]:
    xlsx = data_dir / XLSX_NAME
    if not xlsx.exists():
        raise FileNotFoundError(
            f"CLEAR dataset not found under {data_dir}. Expected "
            f"{XLSX_NAME}. Set --data-dir or CLEAR_DATA_DIR to point at "
            f"the directory."
        )
    df = pd.read_excel(xlsx)
    needed = {"ID", "Excerpt", "BT_easiness"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{xlsx} missing expected column(s): {sorted(missing)}; "
            f"found {sorted(df.columns)}"
        )
    df = df[["ID", "Excerpt", "BT_easiness"]].dropna(
        subset=["ID", "Excerpt", "BT_easiness"]
    )
    return df.to_dict(orient="records")


class ClearData(Data):
    """CLEAR: short reading-passage excerpts, gold = BT_easiness (continuous).

    Single regime; `sample()` draws a population (natural-distribution) n=200
    subsample of the full corpus via an equal-probability random draw.
    """

    def __init__(self, *, data_dir: Path | str | None = None):
        self.data_dir = resolve_clear_dir(data_dir)
        super().__init__(name="clear", context={"dimension": "readability"})

    def load(self) -> Iterable[DataItem]:
        for r in _clear_rows(self.data_dir):
            yield DataItem(
                uid=str(r["ID"]),
                text=str(r["Excerpt"]).strip(),
                gold=float(r["BT_easiness"]),
                extras={
                    "clear_id": str(r["ID"]),
                    "bt_easiness": float(r["BT_easiness"]),
                },
            )

    def sample(self, *, n: int | None = None, seed: int = 0,
               regime: str = "") -> Sample:
        """Equal-probability random subsample of the corpus.

        Deterministic sha256(seed, name, uid) ordering; take the first
        `n` (default `N`=200).
        """
        self._ensure_loaded()
        target = N if n is None else n
        ordered = sorted(
            self._items,
            key=lambda it: hashlib.sha256(
                f"{seed}|{self.name}|{it.uid}".encode()
            ).hexdigest(),
        )
        if len(ordered) < target:
            raise RuntimeError(
                f"CLEAR pool has {len(ordered)} items < requested {target}"
            )
        return Sample(parent=self, items=ordered[:target], regime=regime,
                      seed=seed)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CLEAR loader smoke test")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="override CLEAR data directory")
    args = p.parse_args()
    data = ClearData(data_dir=args.data_dir)
    s = data.sample(seed=42)
    gold = s.gold_array()
    print(f"clear: pool={len(data)}  sample n={len(s)}")
    print(f"  gold (BT_easiness)=[{gold.min():+.3f}, {gold.max():+.3f}]  "
          f"mean={gold.mean():+.3f}  std={gold.std():.3f}")
    print(f"  first 5 uids: {s.uids()[:5]}")
