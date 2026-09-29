"""Merged proxy table: PC (close/far) + global IPP + cyc/mix/eq intransitivity,
per (corpus, judge). Rows come straight from `posbias_close_far_table` and
`transitivity_table`'s `compute_row`, so it cannot disagree with the standalone two.

    uv run proxies/combined_proxy_table.py [--out tables/combined_proxies.tex]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ASAP = REPO / "corpora" / "judge_runs" / "asap_three_proxies"
sys.path.insert(0, str(ASAP))

import posbias_close_far_table as P  # noqa: E402  (PC block + ROWS/gold/edges)
import transitivity_table as TT  # noqa: E402  (cyc/mix/eq block)


def build_records() -> list[dict]:
    """One merged record per (corpus, judge): PC block from `P.compute_row`
    (needs the per-corpus gold-gap edges) + cyc/mix/eq from `TT.compute_row`."""
    gold_cache: dict[str, dict[str, float]] = {}
    edge_cache = {}
    out = []
    for corpus, key, cache_path, name in P.ROWS:
        if key not in gold_cache:
            gold_cache[key] = P.GOLD_LOADERS[key]()
            edge_cache[key] = P._gap_quartile_edges(sorted(gold_cache[key]),
                                                     gold_cache[key])
        gold = gold_cache[key]
        pc = P.compute_row(cache_path, gold, edge_cache[key])
        tri = TT.compute_row(cache_path, gold)
        rec = {**pc, **tri, "corpus": corpus, "judge": name}
        out.append(rec)
    return out


def to_latex(recs: list[dict]) -> str:
    """The bare tabular. Needs multirow + graphicx (rotated corpus label)."""
    n_per_corpus = Counter(r["corpus"] for r in recs)
    lines = [
        r"\begin{tabular}{llcccccccc}",
        r"  \toprule",
        r"  & & \multicolumn{3}{c}{Consistency ($1-$flip)} & IPP & "
        r"\multicolumn{3}{c}{\% non-transitive triads} & \\",
        r"  \cmidrule(lr){3-5}\cmidrule(lr){7-9}",
        r"  & Judge & all & close & far & all & cyc & mix & ineq & $\rho$ \\",
        r"  & & & (Q1 gap) & (Q4 gap) & & & & & \\",
        r"  \midrule",
    ]
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
            f"{P._fmt2(r['pc_all'])} & {P._fmt2(r['pc_close'])} & {P._fmt2(r['pc_far'])} & "
            f"{P._fmt2(r['ipp_all'])} & "
            f"{100 * r['cyc']:.1f} & {100 * r['mix']:.1f} & {100 * r['eq']:.1f} & "
            f"${r['rho']:+.3f}$ \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def text_preview(recs: list[dict]) -> str:
    hdr = (f"{'corpus':<7}{'judge':<16}"
           f"{'PC.all':>9}{'close':>9}{'far':>9}{'IPP.all':>9}"
           f"{'cyc%':>9}{'mix%':>9}{'ineq%':>9}{'ρ':>9}"
           f"   (PC=1-flip; IPP global; non-transitivity cyc->ineq)")
    out = [hdr, "-" * 108]
    for r in recs:
        e = r["edges"]
        out.append(
            f"{r['corpus']:<7}{r['judge']:<16}"
            f"{P._fmt2(r['pc_all']):>9}{P._fmt2(r['pc_close']):>9}{P._fmt2(r['pc_far']):>9}"
            f"{P._fmt2(r['ipp_all']):>9}"
            f"{100 * r['cyc']:>9.1f}{100 * r['mix']:>9.2f}{100 * r['eq']:>9.2f}"
            f"{r['rho']:>+9.3f}"
            f"   close<={e[0]:.2f} far>={e[2]:.2f}  n_tri={r['n_triads']}"
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
