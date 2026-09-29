"""The command line both figure simulations share.

A run that changes a numeric setting has to name itself with --tag.
"""
from __future__ import annotations

import argparse

from .config import (DISAGREE, GOLDS, HUMANS, K_CLOSE, LINKS, N, N_PAIRS,
                     N_RUNS, POPULATIONS, SEED, SHARES, describe_run)
from .naming import shares_changed


def build_parser(description: str, disagree_help: str, k_close_help: str,
                 with_human: bool) -> argparse.ArgumentParser:
    """The options common to both simulations. `with_human` adds --human
    (human-calibrated design only)."""
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--population", choices=POPULATIONS, default="normal")
    if with_human:
        ap.add_argument("--human", choices=HUMANS, default="bt",
                        help="'bt' = the noisy Bradley-Terry human; "
                             "'deterministic' = the higher-ability text always "
                             "wins, so the human is the truth")
    ap.add_argument("--gold", choices=GOLDS, default="human",
                    help="what the ranking is scored against: 'human' = "
                         "Bradley-Terry fitted to the human's own labels (what "
                         "a real evaluation sees), 'true' = the latent ability")
    ap.add_argument("--link", choices=LINKS, default="logistic",
                    help="shape of the human's preference curve: 'logistic' = "
                         "Bradley-Terry (= Gumbel-Max), 'probit' = "
                         "Thurstone-Mosteller, 'cauchy' = heavy tails")
    ap.add_argument("--disagree", type=float, default=DISAGREE,
                    help=disagree_help)
    ap.add_argument("--k-close", dest="k_close", type=float, default=K_CLOSE,
                    help=k_close_help)
    ap.add_argument("--shares", type=str, default=None,
                    help="comma-separated spread shares, e.g. 0,0.5,1")
    ap.add_argument("--items", type=int, default=N)
    ap.add_argument("--pairs", type=int, default=N_PAIRS)
    ap.add_argument("--runs", type=int, default=N_RUNS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", type=str, default=None,
                    help="one short word naming this run, used in the file "
                         "name. Required as soon as a numeric setting differs "
                         "from the default, so numbers stay out of file names.")
    return ap


def require_tag(a: argparse.Namespace) -> None:
    """A run that changes any number has to name itself with --tag."""
    changed = [name for name, differs in (
        ("--disagree", a.disagree != DISAGREE),
        ("--k-close", a.k_close != K_CLOSE),
        ("--shares", shares_changed(a.shares)),
        ("--items", a.items != N),
        ("--pairs", a.pairs != N_PAIRS),
        ("--runs", a.runs != N_RUNS),
        ("--seed", a.seed != SEED),
    ) if differs]
    if changed and not a.tag:
        raise SystemExit(
            "This run changes " + ", ".join(changed) + ". Give it a short "
            "--tag word (e.g. --tag smoketest) so the figure gets a readable "
            "name; the numbers go into the settings JSON.")
    if a.tag and not a.tag.isalnum():
        raise SystemExit("--tag must be a single word of letters or digits")


def resolve_shares(a: argparse.Namespace) -> list[float]:
    shares = ([float(s) for s in a.shares.split(",")] if a.shares
              else list(SHARES))
    if not all(0.0 <= s <= 1.0 for s in shares):
        raise SystemExit("--shares must all be between 0 and 1")
    return shares


def announce(cfg: dict, figure: str) -> None:
    """Print whether this run is the paper figure or a variant."""
    print(describe_run(cfg, figure))
    print("(no arguments = the paper default; --help lists the variants)\n")
