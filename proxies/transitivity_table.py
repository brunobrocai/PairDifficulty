"""Triad-intransitivity table (cyc / mix / eq shares + rho), per (corpus, judge).
Row plan, gold loaders and ALPHA are reused from `posbias_close_far_table`.

    uv run proxies/transitivity_table.py [--out tables/transitivity.tex]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ASAP = REPO / "corpora" / "judge_runs" / "asap_three_proxies"
sys.path.insert(0, str(ASAP))

import posbias_close_far_table as T  # noqa: E402  (same dir; run via `uv run proxies/...`)
from asap_metrics import (  # noqa: E402
    bt_fit,
    load_verdicts,
    spearman,
    triad_intransitivity_swap,
)


def compute_row(cache_path: Path, gold: dict[str, float]) -> dict:
    rows = load_verdicts(cache_path)
    cache_uids = {r["uid_a"] for r in rows} | {r["uid_b"] for r in rows}
    uids = sorted(cache_uids & set(gold))
    g_arr = np.array([gold[u] for u in uids], float)
    rho = spearman(bt_fit(uids, rows, alpha=T.ALPHA), g_arr)
    ti = triad_intransitivity_swap(rows, uids)
    ntri = ti["n_triads"]
    n_cyc, n_eq, n_mix = ti["n_cycle"], ti["n_equality_chain"], ti["n_mixed"]
    pct = lambda n: (n / ntri) if ntri else float("nan")  # noqa: E731
    return {
        "cyc": pct(n_cyc), "mix": pct(n_mix), "eq": pct(n_eq), "rho": rho,
        "n_triads": ntri, "n_cyc": n_cyc, "n_mix": n_mix, "n_eq": n_eq,
    }


def build_records() -> list[dict]:
    gold_cache: dict[str, dict[str, float]] = {}
    out = []
    for corpus, key, cache_path, name in T.ROWS:
        if key not in gold_cache:
            gold_cache[key] = T.GOLD_LOADERS[key]()
        rec = compute_row(cache_path, gold_cache[key])
        rec.update(corpus=corpus, judge=name)
        out.append(rec)
    return out


def to_latex(recs: list[dict]) -> str:
    """The bare tabular. Needs multirow + graphicx (rotated corpus label)."""
    from collections import Counter

    n_per_corpus = Counter(r["corpus"] for r in recs)
    lines = [
        r"\begin{tabular}{clcccc}",
        r"  \toprule",
        r"  & & \multicolumn{3}{c}{\% intransitive triads} & \\",
        r"  \cmidrule(lr){3-5}",
        r"  & Judge & cyc & mix & eq & $\rho$ \\",
        r"  \midrule",
    ]
    last_corpus = None
    for r in recs:
        if r["corpus"] != last_corpus:
            if last_corpus is not None:
                lines.append(r"  \midrule")
            corpus_cell = (rf"\multirow{{{n_per_corpus[r['corpus']]}}}{{*}}"
                           rf"{{\rotatebox[origin=c]{{90}}{{{r['corpus']}}}}}")
        else:
            corpus_cell = ""
        last_corpus = r["corpus"]
        lines.append(
            f"  {corpus_cell} & {r['judge']:<16} & "
            f"{100 * r['cyc']:.1f} & {100 * r['mix']:.1f} & {100 * r['eq']:.1f} & "
            f"${r['rho']:+.3f}$ \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def text_preview(recs: list[dict]) -> str:
    hdr = (f"{'corpus':<7}{'judge':<16}{'cyc%':>7}{'mix%':>7}{'eq%':>7}{'ρ':>9}"
           f"   (cyc/mix/eq counts, n_triads)")
    out = [hdr, "-" * 80]
    for r in recs:
        out.append(
            f"{r['corpus']:<7}{r['judge']:<16}"
            f"{100 * r['cyc']:>7.1f}{100 * r['mix']:>7.2f}{100 * r['eq']:>7.2f}"
            f"{r['rho']:>+9.3f}"
            f"   cyc/mix/eq={r['n_cyc']}/{r['n_mix']}/{r['n_eq']}  n_tri={r['n_triads']}"
        )
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="write the .tex here (also always printed to stdout)")
    args = ap.parse_args(argv)

    recs = build_records()
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
