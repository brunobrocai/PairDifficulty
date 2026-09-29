"""DataItem + Data: the items to judge.

A `DataItem` is one thing the judge looks at:
    uid     str           stable identifier (e.g. essay_id, story-model key)
    text    str           the body of text shown to the judge
    gold    float|None    the gold rating; None if no gold is available
    extras  dict[str,Any] arbitrary per-item fields a prompt might need
                          (e.g. assignment text, model name, rubric label)

A `Data` is a lazily loaded collection of `DataItem`s plus a
`context` dict for things shared across items (e.g. the writing prompt
that is the same for every essay).

To add a corpus, subclass `Data` and implement `load()`. The default
`sample()` orders by sha256 of the uid and takes the first N.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import numpy as np


@dataclass(frozen=True)
class DataItem:
    uid: str
    text: str
    gold: float | None
    extras: dict[str, Any] = field(default_factory=dict)


class Data(ABC):
    """Abstract collection of DataItems.

    Subclass and implement `load()`; optionally override `sample()`.
    `load()` runs the first time the collection is used.
    """

    name: str = ""
    context: dict[str, Any]

    def __init__(self, *, name: str = "", context: dict[str, Any] | None = None):
        self.name = name or type(self).__name__
        self.context = dict(context or {})
        self._items: list[DataItem] = []
        self._by_uid: dict[str, DataItem] = {}
        self._loaded = False

    @abstractmethod
    def load(self) -> Iterable[DataItem]:
        """Yield every DataItem in the pool. Called once, lazily."""

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        for it in self.load():
            self._items.append(it)
            if it.uid in self._by_uid:
                raise ValueError(f"duplicate uid {it.uid!r} in {self.name}")
            self._by_uid[it.uid] = it
        self._loaded = True

    def __iter__(self) -> Iterator[DataItem]:
        self._ensure_loaded()
        return iter(self._items)

    def __getitem__(self, uid: str) -> DataItem:
        self._ensure_loaded()
        return self._by_uid[uid]

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._items)

    def __contains__(self, uid: object) -> bool:
        self._ensure_loaded()
        return uid in self._by_uid

    def uids(self) -> list[str]:
        self._ensure_loaded()
        return [it.uid for it in self._items]

    def items(self) -> list[DataItem]:
        self._ensure_loaded()
        return list(self._items)

    def sample(
        self,
        *,
        n: int | None = None,
        seed: int = 0,
        regime: str = "default",
    ) -> "Sample":
        """Return a deterministic ordered subset.

        Default: sha256-keyed shuffle on uid, take the first `n`. `regime`
        is a label that ends up in the cache name.
        """
        self._ensure_loaded()
        keyed = [
            (
                hashlib.sha256(f"{seed}|{self.name}|{it.uid}".encode()).hexdigest(),
                it,
            )
            for it in self._items
        ]
        keyed.sort(key=lambda x: x[0])
        ordered = [it for _, it in keyed]
        if n is not None:
            ordered = ordered[:n]
        return Sample(
            parent=self, items=ordered, regime=regime, seed=seed,
        )


class Sample:
    """A frozen ordered subset of a Data, with the same iteration / indexing surface.

    Carries regime, seed and parent name for building the cache name.
    """

    def __init__(
        self,
        *,
        parent: Data,
        items: Iterable[DataItem],
        regime: str,
        seed: int,
    ):
        self.parent = parent
        self.items: list[DataItem] = list(items)
        self.regime = regime
        self.seed = seed
        self._by_uid: dict[str, DataItem] = {it.uid: it for it in self.items}

    def __iter__(self) -> Iterator[DataItem]:
        return iter(self.items)

    def __getitem__(self, uid: str) -> DataItem:
        return self._by_uid[uid]

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, uid: object) -> bool:
        return uid in self._by_uid

    def uids(self) -> list[str]:
        return [it.uid for it in self.items]

    def gold_array(self) -> np.ndarray:
        """Gold ratings as a float array aligned with self.uids()."""
        return np.array(
            [float("nan") if it.gold is None else float(it.gold)
             for it in self.items],
            dtype=float,
        )

    def __repr__(self) -> str:
        return (
            f"Sample(parent={self.parent.name!r}, regime={self.regime!r}, "
            f"seed={self.seed}, n={len(self.items)})"
        )
