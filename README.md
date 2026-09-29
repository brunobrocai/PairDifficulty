# Replication package

*Pair Difficulty Matters: Rethinking Pairwise LLM-as-a-Judge Evaluation and Consistency*

This package reproduces the empirical and simulation results of the paper. Pairwise LLM judges are often assessed with position bias and transitivity as logical violations, and with pair-level agreement with human ratings as a gold standard. The paper's claim is that all three are dominated by judge behavior on close-rank-gap pairs (near-ties), where inconsistency is expected and contributes least to the aggregate ranking. So position bias, transitivity, and pair-level agreement correlate only weakly with ranking accuracy against human gold.

The package produces Figures 1 and 2 (simulations), Tables 1 and 2, the significance tests behind the claims in Section 5, and the appendix table for ASAP without gold-tied pairs. Everything runs from the cached judge verdicts shipped here, with no API calls and no GPU. The tables also need each corpus's human gold from its raw dataset, which you supply (see below). The simulations need nothing.

## Requirements

- Python ≥ 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management

```bash
uv sync          # resolve and install the dependencies in pyproject.toml
```

## Reproduce the paper's outputs

```bash
# Figure 1 and Figure 2. No data and no arguments needed.
uv run simulation/human_calibrated_judges.py
uv run simulation/blind_zone_judges.py

# Table 1 — close/far consistency + global IPP + cyc/mix/eq, vs ρ
uv run proxies/combined_proxy_table.py --out tables/combined_proxies.tex

# Table 2 — Spearman of each proxy vs the ranking ρ (Holm-adjusted)
uv run proxies/proxy_predicts_rho_table.py --out-pooled tables/proxy_vs_rho_pooled.tex

# Section 5 significance claims — paired McNemar (flip / IPP), paired Δρ bootstrap
uv run proxies/posbias_close_far_tests.py --out tables/posbias_close_far_tests.md

# Appendix — ASAP rebinned with the gold-tied pairs dropped
uv run proxies/asap_ties_excluded_check.py --out tables/asap_ties_excluded.tex
```

The canonical outputs are also shipped under `tables/` so you can diff your regenerated files against them.

### Off-package inputs: the raw corpora

We ship our verdict caches but no corpus data and no corpus-derived gold. The tables read each corpus's human gold from its raw dataset; see `corpora/PROVENANCE.md` for the official sources:

- **CLEAR** gold (`BT_easiness`) is read from `CLEAR_corpus_final.xlsx`. Place it at `~/Data/LLM-as-judge/CLEAR/CLEAR_corpus_final.xlsx`, or set `CLEAR_XLSX=/path/to/CLEAR_corpus_final.xlsx`.
- **ASAP** gold (FACS holistic 1–6) is regenerated from the raw ASAP 2.0 CSV by a deterministic seed-42 sample: the 200 grade-10 essays on the *Facial action coding system* prompt, drawn to a quota that matches the natural FACS score distribution. Place `ASAP_2_Final_github_test.csv` under `~/Data/LLM-as-judge/ASAP2/`, or set `ASAP2_DATA_DIR=/path/to/dir`.

If a dataset is missing, the run stops with a `FileNotFoundError` naming the path it looked for.

## Layout

```
simulation/
  human_calibrated_judges.py  -> Figure 1 (results/ is created on run). Also
                              writes sim_humancalib_sweep.png, the full sweep
                              behind it; not printed in the paper.
  blind_zone_judges.py        -> Figure 2, plus its own sweep companion.
  btsim/                      everything the two figures share: the
                              Bradley-Terry fit and judge simulation (bt.py),
                              the population draw (population.py), the settings
                              (config.py), the table and plots (plots.py), the
                              file naming (naming.py), the command line
                              (cli.py), the preference curves (links.py) and
                              the colour scheme (style.py)

proxies/
  combined_proxy_table.py     -> Table 1
  proxy_predicts_rho_table.py -> Table 2
  posbias_close_far_tests.py  -> the Section 5 significance claims (markdown)
  posbias_close_far_table.py  consistency/IPP/ρ engine, gold loaders, row plan
                              (shared)
  transitivity_table.py       cyc/mix/eq triad engine (shared)
  posbias_intransitivity_absorption.py  posterior-predictive BT diagnostic

corpora/judge_runs/
  clear/               CLEAR readability pairwise pipeline + 6-judge verdict caches
  asap_three_proxies/  ASAP 2.0 FACS pairwise pipeline + metrics + caches
                       (gold regenerated from the raw corpus, seed 42)

src/core/              shared LLM-judge engine (judges, methods, cache, prompts)
preregistration/       the sampling/master seeds the pipelines import
tables/                the canonical generated outputs, shipped as reference
```

The six judges behind both tables are GPT-5.4-nano, GPT-5.4-mini, Ministral-3-3B, Ministral-3-14B, Gemma-3-4B, and Gemma-3-12B.

## Judging setup (how the verdicts were collected)

For each corpus we draw one sample of n = 200 items and judge a soft-regular schedule of 1200 unordered pairs, i.e. 12 comparisons per item. Bradley–Terry ranking quality levels off at around this number. Every pair is judged in both orders, (A,B) and (B,A), for 2400 verdicts in total. Judges run at temperature 0, reasoning effort `none`, seed 42, prompt version `v1`, and `soft_x = 64`. These settings are recorded in each cache's `manifest_*.json`.

## Regenerating the judge caches (optional, needs API / GPU)

The full pipelines are included so the verdicts can be re-collected. This needs the raw corpora (see `corpora/PROVENANCE.md`) and either an OpenAI key (for the GPT judges, put `OPENAI_API_KEY` in a `.env` at the package root) or local model access for the open-weight judges (`uv sync --extra local` or `--extra cluster`).

```bash
uv run corpora/judge_runs/clear/run.py            pairwise --judge gpt-5.4-nano
uv run corpora/judge_runs/asap_three_proxies/run.py pairwise --judge gpt-5.4-nano
```

Runs resume from the existing caches by `call_id`; the sample (seed 42) and the soft-regular schedule (`--budget 1200 --soft-x 64`) are pinned, so a fresh run reproduces the same comparisons.

## What the caches contain

Each cache line is a single verdict, `{uid_a, uid_b, verdict ∈ {A,B}}`, keyed by item id. No corpus text is included.

## Citation

Forthcoming.

## License

Code in this package is released under the **MIT License** (see `LICENSE`). The judge verdict caches are the authors' own measurements, released under **CC-BY-4.0**. The third-party text corpora (CLEAR, ASAP 2.0) are **not** included and remain under their original licenses.

## Use of AI

Claude Code (Anthropic) — with Claude Opus 4.6, Opus 4.7, Opus 5 and Opus 5.5 at high and extra-high reasoning effort — was used to assist with the code in this package: implementation, refactoring, repository cleanup, and reproducibility checks.
