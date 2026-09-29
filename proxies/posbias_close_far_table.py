"""Position-bias close/far table (PC = 1-flip, IPP = primacy-flip rate), per
(corpus, judge), bucketed by gold gap.

    uv run proxies/posbias_close_far_table.py [--out tables/posbias_close_far.tex]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ASAP = REPO / "corpora" / "judge_runs" / "asap_three_proxies"
sys.path.insert(0, str(ASAP))
from asap_metrics import (  # noqa: E402
    _unordered_pair_key,
    bt_fit,
    load_verdicts,
    spearman,
)

# Judge caches live in-repo; the raw CLEAR corpus is third-party and off-repo.
CLEAR_CACHE_DIR = Path(
    os.getenv("CLEAR_CACHE_DIR",
              REPO / "corpora" / "judge_runs" / "clear" / "cache" / "pairwise")
)
CLEAR_XLSX = Path(
    os.getenv("CLEAR_XLSX",
              Path.home() / "Data" / "LLM-as-judge" / "CLEAR"
              / "CLEAR_corpus_final.xlsx")
)

# Row plan: (corpus, gold_loader_key, cache_path, display_name).
ASAP_PW = ASAP / "cache" / "pairwise"
ROWS = [
    ("CLEAR", "clear", CLEAR_CACHE_DIR / "cache_gpt-5_4-nano_v1.jsonl",
     "GPT-5.4-nano"),
    ("CLEAR", "clear", CLEAR_CACHE_DIR / "cache_gpt-5_4-mini_v1.jsonl",
     "GPT-5.4-mini"),
    ("CLEAR", "clear", CLEAR_CACHE_DIR / "cache_mistralai-Ministral-3-3B-Instruct-2512_v1.jsonl",
     "Ministral-3-3B"),
    ("CLEAR", "clear", CLEAR_CACHE_DIR / "cache_mistralai-Ministral-3-14B-Instruct-2512_v1.jsonl",
     "Ministral-3-14B"),
    ("CLEAR", "clear", CLEAR_CACHE_DIR / "cache_google-gemma-3-4b-it_v1.jsonl",
     "Gemma-3-4B"),
    ("CLEAR", "clear", CLEAR_CACHE_DIR / "cache_google-gemma-3-12b-it_v1.jsonl",
     "Gemma-3-12B"),
    ("ASAP", "asap", ASAP_PW / "cache_gpt-5_4-nano_v1.jsonl",
     "GPT-5.4-nano"),
    ("ASAP", "asap", ASAP_PW / "cache_gpt-5_4-mini_v1.jsonl",
     "GPT-5.4-mini"),
    ("ASAP", "asap", ASAP_PW / "cache_mistralai-Ministral-3-3B-Instruct-2512_v1.jsonl",
     "Ministral-3-3B"),
    ("ASAP", "asap", ASAP_PW / "cache_mistralai-Ministral-3-14B-Instruct-2512_v1.jsonl",
     "Ministral-3-14B"),
    ("ASAP", "asap", ASAP_PW / "cache_google-gemma-3-4b-it_v1.jsonl",
     "Gemma-3-4B"),
    ("ASAP", "asap", ASAP_PW / "cache_google-gemma-3-12b-it_v1.jsonl",
     "Gemma-3-12B"),
]

ALPHA = 0.01


# --- Gold loaders ---

def _gold_asap() -> dict[str, float]:
    # Regenerated from the raw ASAP 2.0 corpus (third-party, off-package),
    # mirroring CLEAR's live read of its xlsx. The loader's deterministic
    # seed-42 population-quota sample reproduces the exact n=200
    # uid -> FACS holistic gold map. Raw CSV path: --data-dir / ASAP2_DATA_DIR
    # / ~/Data/LLM-as-judge/ASAP2 (see corpora/PROVENANCE.md).
    sys.path.insert(0, str(REPO / "src"))   # data.py imports core.*
    from data import ASAP2FacsData          # ASAP dir already on sys.path
    sample = ASAP2FacsData().sample(seed=42)
    return {it.uid: float(it.extras["gold_score"]) for it in sample.items}


def _gold_clear() -> dict[str, float]:
    import pandas as pd
    df = pd.read_excel(CLEAR_XLSX, usecols=["ID", "BT_easiness"]).dropna()
    return {str(r.ID): float(r.BT_easiness) for r in df.itertuples()}


GOLD_LOADERS = {"asap": _gold_asap, "clear": _gold_clear}


# --- Per-row computation ---

def _gap_quartile_edges(uids: list[str], gold: dict[str, float]) -> np.ndarray:
    """Q1/median/Q3 of |g_a − g_b| over all C(n,2) pairs (judge-independent)."""
    gaps = [abs(gold[a] - gold[b]) for a, b in itertools.combinations(uids, 2)]
    return np.quantile(gaps, [0.25, 0.50, 0.75])


def _ipp(rows, gold, *, pred=None) -> tuple[float, int]:
    """Inconsistent-Pair Primacy: among flips, the fraction toward the first
    slot in both orders. Returns (ipp, n_flips); `pred(gap)` buckets by gold gap."""
    by_pair: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        a, b, v = r.get("uid_a"), r.get("uid_b"), r.get("verdict")
        if a not in gold or b not in gold or v not in ("A", "B"):
            continue
        if pred is not None and not pred(abs(gold[a] - gold[b])):
            continue
        by_pair[_unordered_pair_key(a, b)][a].append(a if v == "A" else b)
    n_flip = n_prim = 0
    for by_first in by_pair.values():
        if len(by_first) < 2:
            continue  # only one slot order present
        dir_winners = {first: Counter(ws).most_common(1)[0][0]
                       for first, ws in by_first.items()}
        if len(set(dir_winners.values())) < 2:
            continue  # consistent — not a flip
        n_flip += 1
        # primacy flip <=> in every order the winner is the *first* slot, i.e.
        # always the uid that sat first.
        n_prim += all(win == first for first, win in dir_winners.items())
    return (n_prim / n_flip if n_flip else float("nan")), n_flip


def _flip(rows, gold, *, pred=None) -> tuple[float, int]:
    """Order-swap flip rate over both-order pairs (PC = 1 - flip_rate). Returns
    (flip_rate, n_both); `pred(gap)` buckets by gold gap (same as `_ipp`)."""
    by_pair: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        a, b, v = r.get("uid_a"), r.get("uid_b"), r.get("verdict")
        if a not in gold or b not in gold or v not in ("A", "B"):
            continue
        if pred is not None and not pred(abs(gold[a] - gold[b])):
            continue
        by_pair[_unordered_pair_key(a, b)][a].append(a if v == "A" else b)
    n_both = n_flip = 0
    for by_first in by_pair.values():
        if len(by_first) < 2:
            continue  # only one slot order present
        dir_winners = {first: Counter(ws).most_common(1)[0][0]
                       for first, ws in by_first.items()}
        n_both += 1
        n_flip += int(len(set(dir_winners.values())) > 1)
    return (n_flip / n_both if n_both else float("nan")), n_both


def compute_row(cache_path: Path, gold: dict[str, float], edges: np.ndarray,
                *, integer_diff_buckets: bool = False) -> dict:
    rows = load_verdicts(cache_path)
    cache_uids = {r["uid_a"] for r in rows} | {r["uid_b"] for r in rows}
    uids = sorted(cache_uids & set(gold))
    g_arr = np.array([gold[u] for u in uids], float)
    rho = spearman(bt_fit(uids, rows, alpha=ALPHA), g_arr)
    if integer_diff_buckets:
        # Discrete-gold buckets that exclude gold-tied pairs (gap 0): the
        # closest *distinct* pairs (gap == 1) vs. clearly-separated (gap >= 2).
        is_close = lambda d: d == 1  # noqa: E731
        is_far = lambda d: d >= 2    # noqa: E731
    else:
        # inclusive bounds: discrete gold puts the Q1 edge on a tie value, so `<`
        # would drop the closest pairs.
        is_close = lambda d: d <= edges[0]  # noqa: E731
        is_far = lambda d: d >= edges[2]    # noqa: E731
    ipp_all, ni_all = _ipp(rows, gold)
    ipp_close, ni_close = _ipp(rows, gold, pred=is_close)
    ipp_far, ni_far = _ipp(rows, gold, pred=is_far)
    f_all, nb_all = _flip(rows, gold)
    f_close, nb_close = _flip(rows, gold, pred=is_close)
    f_far, nb_far = _flip(rows, gold, pred=is_far)
    pc = lambda f: (1.0 - f) if f == f else float("nan")  # noqa: E731
    return {
        "ipp_all": ipp_all, "ipp_close": ipp_close, "ipp_far": ipp_far,
        "pc_all": pc(f_all), "pc_close": pc(f_close), "pc_far": pc(f_far),
        "flip": f_all, "rho": rho,
        "ni_all": ni_all, "ni_close": ni_close, "ni_far": ni_far,
        "nb_all": nb_all, "nb_close": nb_close, "nb_far": nb_far,
        "edges": edges, "integer_diff_buckets": integer_diff_buckets,
    }


# --- Rendering ---

def build_records(*, asap_diff_buckets: bool = False) -> list[dict]:
    """Compute one record per (corpus, judge) row.

    `asap_diff_buckets` swaps the ASAP close/far definition from gap quartiles
    to integer-gap buckets (close = gap 1, far = gap >= 2), which excludes the
    gold-tied (gap 0) pairs that otherwise dominate the ASAP close quartile.
    CLEAR gold is continuous, so it always uses the quartile buckets.
    """
    gold_cache: dict[str, dict[str, float]] = {}
    edge_cache: dict[str, np.ndarray] = {}
    out = []
    for corpus, key, cache_path, name in ROWS:
        if key not in gold_cache:
            gold_cache[key] = GOLD_LOADERS[key]()
        gold = gold_cache[key]
        if key not in edge_cache:
            uids = sorted(gold)
            edge_cache[key] = _gap_quartile_edges(uids, gold)
        integer_diff = asap_diff_buckets and corpus == "ASAP"
        rec = compute_row(cache_path, gold, edge_cache[key],
                          integer_diff_buckets=integer_diff)
        rec.update(corpus=corpus, judge=name)
        out.append(rec)
    return out


def _fmt2(x: float) -> str:
    """A rate in [0,1] to two decimals with the leading zero stripped (.78);
    `--` for NaN (e.g. an IPP bucket with no flips)."""
    if x != x:
        return "--"
    s = f"{x:.2f}"
    return s[1:] if s.startswith("0") else s


def to_latex(recs: list[dict]) -> str:
    """The bare tabular. Needs multirow + graphicx (rotated corpus label)."""
    n_per_corpus = Counter(r["corpus"] for r in recs)
    # When ASAP uses integer-gap buckets the close/far definition differs by
    # corpus, so the hard-coded "(Q1 gap)/(Q4 gap)" subheader would be wrong for
    # ASAP; drop it and leave the bucket spec to the manually-added caption.
    mixed_buckets = any(r.get("integer_diff_buckets") for r in recs)
    lines = [
        r"\begin{tabular}{llccccccc}",
        r"  \toprule",
        r"  & & \multicolumn{3}{c}{Consistency ($1-$flip)} & "
        r"\multicolumn{3}{c}{IPP (primacy-flip rate)} & \\",
        r"  \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"  & Judge & all & close & far & all & close & far & $\rho$ \\",
    ]
    if not mixed_buckets:
        lines.append(
            r"  & & & (Q1 gap) & (Q4 gap) & & (Q1 gap) & (Q4 gap) & \\")
    lines.append(r"  \midrule")
    last_corpus = None
    for r in recs:
        if r["corpus"] != last_corpus:
            if last_corpus is not None:
                lines.append(r"  \midrule")  # group separator between corpora
            corpus_cell = (rf"\multirow{{{n_per_corpus[r['corpus']]}}}{{*}}"
                           rf"{{\rotatebox[origin=c]{{90}}{{{r['corpus']}}}}}")
        else:
            corpus_cell = ""  # spanned by the group's \multirow
        last_corpus = r["corpus"]

        lines.append(
            f"  {corpus_cell} & {r['judge']:<16} & "
            f"{_fmt2(r['pc_all'])} & {_fmt2(r['pc_close'])} & {_fmt2(r['pc_far'])} & "
            f"{_fmt2(r['ipp_all'])} & {_fmt2(r['ipp_close'])} & {_fmt2(r['ipp_far'])} & "
            f"${r['rho']:+.3f}$ \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def text_preview(recs: list[dict]) -> str:
    hdr = (f"{'corpus':<7}{'judge':<16}"
           f"{'PC.all':>9}{'close':>9}{'far':>9}"
           f"{'IPP.all':>9}{'close':>9}{'far':>9}{'ρ':>9}"
           f"   (PC=1-flip; IPP=primacy dir among flips)")
    out = [hdr, "-" * 100]
    for r in recs:
        e = r["edges"]
        if r.get("integer_diff_buckets"):
            bucket = "close=gap1 far>=2"
        else:
            bucket = f"close<={e[0]:.2f} far>={e[2]:.2f}"
        out.append(
            f"{r['corpus']:<7}{r['judge']:<16}"
            f"{_fmt2(r['pc_all']):>9}{_fmt2(r['pc_close']):>9}{_fmt2(r['pc_far']):>9}"
            f"{_fmt2(r['ipp_all']):>9}{_fmt2(r['ipp_close']):>9}{_fmt2(r['ipp_far']):>9}"
            f"{r['rho']:>+9.3f}"
            f"   {bucket}"
            f"  n_flips(all/close/far)={r['ni_all']}/{r['ni_close']}/{r['ni_far']}"
        )
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="write the .tex here (also always printed to stdout)")
    ap.add_argument("--asap-diff-buckets", action="store_true",
                    help="for ASAP, use integer-gap buckets (close=gap 1, "
                         "far=gap>=2) that exclude gold-tied pairs, instead of "
                         "gap quartiles")
    args = ap.parse_args(argv)

    recs = build_records(asap_diff_buckets=args.asap_diff_buckets)
    print(text_preview(recs))
    print()
    tex = to_latex(recs)
    print(tex)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(tex + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
