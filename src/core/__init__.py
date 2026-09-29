"""Shared primitives for pointwise-vs-pairwise LLM-judging pilots.

Public surface:
    core.data          — DataItem, Data (subclass per pilot)
    core.cache         — CacheStore (JSONL + manifest)
    core.prompts       — Prompt, PromptRegistry (YAML-backed, hash-verified)
    core.llm_client    — get_client()
    core.methods       — BaseRun, PairwiseRun, PairwiseSoftRun
    core.cli           — make_run_parser, make_analyze_parser
"""
