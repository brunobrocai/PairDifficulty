# Data provenance

This study uses two third-party text corpora as substrates. **Neither corpus is redistributed in this package, and no corpus-derived gold is shipped.** What ships here is the judges' own pairwise verdict caches. To recompute either corpus's gold (needed for the empirical tables) or to regenerate the caches, obtain the raw corpora from their official sources.

## CLEAR (readability)

- **What it is:** the CommonLit CLEAR corpus — English passages with a continuous readability-ease gold, `BT_easiness` (higher = easier), fit by Bradley–Terry from teachers' pairwise "which is easier to read" judgments.
- **Obtain:** the official CLEAR corpus spreadsheet (`CLEAR_corpus_final.xlsx`), publicly released by CommonLit. The analysis reads only the `ID` and `BT_easiness` columns.
- **Where to put it:** `~/Data/LLM-as-judge/CLEAR/CLEAR_corpus_final.xlsx`, or set `CLEAR_XLSX=/path/to/CLEAR_corpus_final.xlsx`.
- **Sample:** one population sample of n=200 passages at the corpus's natural easiness distribution.

## ASAP 2.0 (student essays)

- **What it is:** the ASAP 2.0 student-essay corpus, FACS slice; human gold is the holistic score (1–6).
- **Obtain:** the raw ASAP 2.0 CSV (`ASAP_2_Final_github_test.csv`) from the official ASAP 2.0 release.
- **Where to put it:** `~/Data/LLM-as-judge/ASAP2/`, or set `ASAP2_DATA_DIR`.
- **Gold (regenerated, not shipped):** the table reads the `uid -> holistic gold` map produced by `asap_three_proxies/data.py`'s deterministic seed-42 population-quota n=200 sample of the raw corpus (fixed `sha256(seed, name, uid)` ordering + a fixed score quota, so it reproduces identically every run). No gold file is shipped — the raw CSV above is required, exactly as CLEAR requires its xlsx.

## Judge verdict caches

Under each corpus's `cache/pairwise/`, the `cache_<judge>_v1.jsonl` files are this study's own measurements: one line per comparison, `{uid_a, uid_b, verdict ∈ {A, B}, ...}`, keyed by item id — **no corpus text is embedded**. Pairwise runs are swap-augmented (each unordered pair judged in both orders). Six judges per corpus: GPT-5.4-nano, GPT-5.4-mini, Ministral-3-3B, Ministral-3-14B, Gemma-3-4B, Gemma-3-12B. The accompanying `manifest_`, `ranking_`, and `schedule_` JSON files are run metadata (seed, prompt hash, budget, the fitted BT ranking, the comparison schedule).
