"""argparse + RunConfig helpers for the per-corpus run.py scripts.

`--analyze-after <subcmd>` runs the corpus's analyze.py with that
subcommand after a successful run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from core.methods.base import RunConfig, RunResult
from preregistration.seeds import MASTER_SEED, SAMPLE_SEED


REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high")


def add_common_run_flags(
    p: argparse.ArgumentParser,
    *,
    regimes: Sequence[str] = (),
    default_judge: str = "gpt-4o-mini",
    default_temperature: float = 0.0,
    default_reasoning_effort: str = "none",
    default_prompt_version: str = "v1",
    require_regime: bool = True,
) -> None:
    """Add the run-flags shared by every method.

    `regimes=()` skips the --regime flag."""
    p.add_argument("--judge", default=default_judge)
    if regimes:
        p.add_argument(
            "--regime",
            required=require_regime,
            choices=list(regimes),
            help=f"sampling regime: one of {tuple(regimes)}",
        )
    p.add_argument(
        "--n", type=int, default=None,
        help="sample size; pilots with fixed-quota regimes may ignore",
    )
    p.add_argument(
        "--sample-seed", type=int, default=SAMPLE_SEED,
        help=f"default {SAMPLE_SEED} per preregistration/seeds.md "
             f"(locked single-seed value). Override only for debug/smoke; "
             f"reported numbers must use the preregistered value.",
    )
    p.add_argument(
        "--master-seed", type=int, default=MASTER_SEED,
        help=f"default {MASTER_SEED} per preregistration/seeds.md "
             f"(locked single-seed value). Override only for debug/smoke; "
             f"reported numbers must use the preregistered value.",
    )
    p.add_argument("--temperature", type=float, default=default_temperature)
    p.add_argument(
        "--reasoning-effort",
        default=default_reasoning_effort,
        choices=REASONING_EFFORTS,
    )
    p.add_argument(
        "--prompt-version", default=default_prompt_version,
        help="prompt version to load from the pilot's prompts/ dir",
    )
    p.add_argument(
        "--cache", type=Path, default=None,
        help="override the cache file path (default: derived from "
             "judge + version + regime + tag)",
    )
    p.add_argument("--tag", default="", help="extra suffix on cache stem")
    p.add_argument(
        "--limit", type=int, default=None,
        help="cap on number of LLM calls (smoke test)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="print the first prompt without issuing any calls",
    )
    p.add_argument(
        "--retry-failed", action="store_true",
        help="re-attempt cached rows whose response_field is null (transient "
             "API failures: rate-limit, timeout, schema reject). Successful "
             "rows are kept; failed call_ids are re-queried and the new "
             "result is appended (load_keyed is last-write-wins).",
    )
    p.add_argument(
        "--analyze-after", default=None,
        help="after a successful run, invoke analyze.py with this subcommand",
    )


def add_pairwise_flags(p: argparse.ArgumentParser, default_degree: int = 10) -> None:
    """Add the extra flags pairwise needs."""
    p.add_argument(
        "--degree", type=int, default=default_degree,
        help="BASE d-regular degree (= comparisons per item on the first "
             "run). Locked once a cache exists; bump via --extend-by.",
    )
    p.add_argument(
        "--extend-by", type=int, default=0,
        help="add this many *new* comparisons per item, disjoint from "
             "the prior schedule. Requires an existing cache. Each "
             "invocation appends one entry to the manifest's "
             "extension_history.",
    )
    p.add_argument(
        "--alpha", type=float, default=0.01,
        help="BT L2 regularization alpha (default 0.01; bt_fit raises if "
             "ILSR doesn't converge — collect more comparisons rather than "
             "raising alpha unless you've decided a stronger prior is justified)",
    )


def add_pairwise_soft_flags(
    p: argparse.ArgumentParser,
    *,
    default_soft_x: float = 1024.0,
    default_budget: int | None = None,
) -> None:
    """Flags for the soft-regular pairwise scheduler (PairwiseSoftRun).
    `--budget` is the total pair count; `--soft-x` is the uniformity
    exponent (default 1024, effectively d-regular). `--budget` is required
    unless `default_budget` is given."""
    if default_budget is None:
        p.add_argument(
            "--budget", type=int, required=True,
            help="number of unique pairs in the schedule. Grow-only on resume "
                 "(bumping it adds new pairs disjoint from the prior run; "
                 "shrinking it raises).",
        )
    else:
        p.add_argument(
            "--budget", type=int, default=default_budget,
            help=f"number of unique pairs in the schedule (default "
                 f"{default_budget}). Grow-only on resume (bumping it adds new "
                 f"pairs disjoint from the prior run; shrinking it raises).",
        )
    p.add_argument(
        "--soft-x", type=float, default=default_soft_x,
        help=f"uniformity exponent for the soft scheduler (default "
             f"{default_soft_x}; locked once a cache exists). Higher = "
             f"more uniform; at X≥256 the schedule is exactly per-item "
             f"degree-regular at wave boundaries.",
    )
    p.add_argument(
        "--alpha", type=float, default=0.01,
        help="BT L2 regularization alpha (default 0.01; raise budget rather "
             "than alpha if ILSR doesn't converge)",
    )


def config_from_args(
    args: argparse.Namespace,
    *,
    prompt_key: str,
    extra: dict | None = None,
) -> RunConfig:
    """Build a RunConfig from a Namespace produced by `add_common_run_flags`."""
    return RunConfig(
        judge=args.judge,
        prompt_key=prompt_key,
        prompt_version=args.prompt_version,
        sample_seed=args.sample_seed,
        master_seed=args.master_seed,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        tag=args.tag,
        limit=args.limit,
        dry_run=args.dry_run,
        retry_failed=args.retry_failed,
        extra=extra or {},
    )


def maybe_chain_analyze(
    args: argparse.Namespace,
    *,
    analyze_script: Path,
    result: RunResult,
) -> None:
    """If --analyze-after was set, invoke analyze.py with that subcommand.

    Passes through --judge, --regime, --tag and --prompt-version.
    """
    if not args.analyze_after or result.dry_run:
        return
    if result.made == 0 and result.skipped == 0:
        print("[skip analyze] no completed calls.")
        return
    cmd = [
        sys.executable, str(analyze_script), args.analyze_after,
        "--judge", args.judge,
        "--prompt-version", args.prompt_version,
        "--sample-seed", str(args.sample_seed),
    ]
    regime = getattr(args, "regime", None)
    if regime:
        cmd.extend(["--regime", regime])
    if args.n is not None:
        cmd.extend(["--n", str(args.n)])
    if args.tag:
        cmd.extend(["--tag", args.tag])
    print(f"\n[analyze-after] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# --- Analyze side ---

def add_common_analyze_flags(
    p: argparse.ArgumentParser,
    *,
    regimes: Sequence[str] = (),
    default_judge: str = "gpt-4o-mini",
    default_prompt_version: str = "v1",
    default_alpha: float = 0.01,
) -> None:
    """Flags every analyze subcommand needs to locate its cache.

    `regimes=()` skips --regime. `--alpha` is the BT L2 regularizer."""
    p.add_argument("--judge", default=default_judge)
    if regimes:
        p.add_argument(
            "--regime", required=True, choices=list(regimes),
        )
    p.add_argument("--prompt-version", default=default_prompt_version)
    p.add_argument("--tag", default="")
    p.add_argument(
        "--alpha", type=float, default=default_alpha,
        help="BT L2 regularization alpha (default 0.01; bt_fit raises if "
             "ILSR doesn't converge — collect more comparisons rather than "
             "raising alpha unless you've decided a stronger prior is justified)",
    )
