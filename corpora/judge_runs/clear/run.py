"""CLEAR final-eval runner: swap-augmented pairwise judging. Single construct
(no --dim). Population sample n=200; soft-regular scheduler (--soft-x 64);
--budget default 1200 (12 comparisons/text). Resumes from cache by call_id.

    uv run corpora/judge_runs/clear/run.py pairwise --judge gpt-5.4-nano [--budget 1200] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Path bootstrap: src/ for `core.*`, repo root for `preregistration.*`, this
# dir for local `prompts`/`data` (corpora/ is not a package).
_HERE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _HERE_DIR.parents[2]  # corpora/judge_runs/clear -> repo root
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_HERE_DIR))

from core.cli import (
    add_common_run_flags, add_pairwise_soft_flags,
    config_from_args,
)
from core.judges import make_judge
from core.methods.pairwise_soft import PairwiseSoftRun

import prompts  # noqa: F401 — registers the pairwise prompt key
from data import ClearData

_HERE = Path(__file__).resolve().parent
_CACHE_ROOT = _HERE / "cache"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clear-run",
        description="CLEAR final-eval: swap-augmented pairwise judging.",
    )
    sub = p.add_subparsers(dest="method", required=True)

    # budget default 1200 = 12 comparisons/text at n=200 (12*200/2),
    # the budget used in the paper.
    pr = sub.add_parser("pairwise")
    _add_data_dir_flag(pr)
    add_common_run_flags(pr, default_judge="gpt-5.4-nano")
    add_pairwise_soft_flags(pr, default_soft_x=64.0, default_budget=1200)
    return p


def _add_data_dir_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data-dir", type=Path, default=None,
        help="CLEAR data directory (must contain CLEAR_corpus_final.xlsx). "
             "Falls back to $CLEAR_DATA_DIR, then to "
             "~/Data/LLM-as-judge/CLEAR.",
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.n is not None:
        raise SystemExit(
            "--n is not supported for CLEAR: the population (natural-"
            "distribution) sample is fixed at n=200 (locked in data.py). "
            "Drop the flag; use --limit to cap calls for a smoke test."
        )

    data = ClearData(data_dir=args.data_dir)
    sample = data.sample(seed=args.sample_seed)

    prompt_key = f"clear.{args.method}"
    cfg = config_from_args(args, prompt_key=prompt_key)
    cache_root = _CACHE_ROOT / args.method

    runner = PairwiseSoftRun(
        data=data, sample=sample, cfg=cfg, cache_root=cache_root,
        budget=args.budget, soft_x=args.soft_x, alpha=args.alpha,
    )

    judge = None if args.dry_run else make_judge(args.judge)
    runner.execute(judge)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
