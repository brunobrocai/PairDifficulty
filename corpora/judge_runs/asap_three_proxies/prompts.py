"""Register the ASAP 2.0 FACS pairwise prompt (asap2_facs.pairwise — pick the
higher-quality of two essays, A/B) from prompts/pairwise/v*.yaml.
"""

from __future__ import annotations

from pathlib import Path

from core.prompts import register_dir

_HERE = Path(__file__).resolve().parent

register_dir("asap2_facs.pairwise", _HERE / "prompts" / "pairwise")
