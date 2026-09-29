"""ASAP 2.0 FACS final-eval runner: swap-augmented pairwise judging. Single
dimension (no --dim). Population-quota sample n=200; soft-regular scheduler
(--soft-x 64); --budget default 1200 (12 comparisons/text). The raw corpus is
third-party and off-repo: the loader reads ASAP_2_Final_github_test.csv from
--data-dir, else $ASAP2_DATA_DIR, else ~/Data/LLM-as-judge/ASAP2. Resumes from
cache by call_id.

    uv run corpora/judge_runs/asap_three_proxies/run.py pairwise --judge gpt-5.4-nano [--budget 1200] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Path bootstrap: src/ for `core.*`, repo root for `preregistration.*`, this
# dir for local `prompts`/`data` (corpora/ is not a package).
_HERE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _HERE_DIR.parents[2]  # corpora/judge_runs/asap_three_proxies -> repo root
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
from data import ASAP2FacsData

_HERE = Path(__file__).resolve().parent
_CACHE_ROOT = _HERE / "cache"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="asap2-facs-run",
        description="ASAP2 FACS final-eval: swap-augmented pairwise judging.",
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
        help="ASAP2 data directory (must contain "
             "ASAP_2_Final_github_test.csv). Falls back to $ASAP2_DATA_DIR, "
             "then to ~/Data/LLM-as-judge/ASAP2.",
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.n is not None:
        raise SystemExit(
            "--n is not supported for ASAP2 FACS: the population-quota "
            "sample is fixed at n=200. Drop the flag."
        )

    data = ASAP2FacsData(data_dir=args.data_dir)
    sample = data.sample(seed=args.sample_seed)

    prompt_key = f"asap2_facs.{args.method}"
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
