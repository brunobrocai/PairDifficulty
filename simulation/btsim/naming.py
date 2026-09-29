"""How a run's settings become a file name.

Both figure scripts write `<stem>_main<suffix>.png`, `<stem>_sweep<suffix>.png`
and `<stem>_settings<suffix>.json`. Only word-shaped settings go into the
suffix; anything numeric has to be named with a --tag, so a file name never
carries a number that could be misread.
"""
from __future__ import annotations

from .config import SHARES


def suffix(population: str, tag: str | None, link: str = "logistic",
           gold: str = "human", human: str = "bt") -> str:
    """Words only. Numeric settings (flipped share, concentration) must be
    named with a --tag instead of being spelled out in the file name."""
    bits = []
    if population != "normal":
        bits.append(population)
    if human != "bt":
        bits.append("oraclehuman")
    if link != "logistic":
        bits.append(link)
    if gold != "human":
        bits.append("truegold")
    if tag:
        bits.append(tag)
    return ("_" + "_".join(bits)) if bits else ""


def shares_changed(raw: str | None) -> bool:
    if raw is None:
        return False
    return [float(x) for x in raw.split(",")] != list(SHARES)
