"""Register the CLEAR pairwise prompt (clear.pairwise — pick the easier of two
texts, A/B) from prompts/pairwise/v*.yaml.
"""

from __future__ import annotations

from pathlib import Path

from core.prompts import register_dir

_HERE = Path(__file__).resolve().parent

register_dir("clear.pairwise", _HERE / "prompts" / "pairwise")
