"""Judge factory: name -> Judge instance.

Construct judges via `make_judge(name)`; routing lives here in one place.

- Names in `_LOCAL_JUDGES` (open-weight models) route to `LlamaCppJudge`
  on the default Q8_0 GGUF path, or to `TransformersJudge` (BF16) when
  `JUDGE_BACKEND=hf` is set — the cluster path for the larger models.
- Any other name routes to `OpenAIJudge` (the name is the OpenAI model
  id). The paper's GPT judges are gpt-5.4-nano / gpt-5.4-mini.
"""

from __future__ import annotations

import os

from core.judges.base import Judge
from core.judges.llamacpp_judge import LlamaCppJudge
from core.judges.openai_judge import OpenAIJudge


# Per-model overrides for local judges, keyed by canonical name. Values
# are kwargs forwarded to the Judge constructor; anything not listed here
# falls through to OpenAIJudge. `hf_repo`/`hf_filename` give the canonical
# Q8_0 GGUF (auto-downloaded when no GGUF_<NAME>_PATH is set);
# `disable_thinking` turns off Qwen3's thinking block.
_LOCAL_JUDGES: dict[str, dict] = {
    "Qwen/Qwen3-0.6B": {
        "disable_thinking": True,
        "hf_repo": "Qwen/Qwen3-0.6B-GGUF",
        "hf_filename": "Qwen3-0.6B-Q8_0.gguf",
    },
    "Qwen/Qwen3-1.7B": {
        "disable_thinking": True,
        "hf_repo": "Qwen/Qwen3-1.7B-GGUF",
        "hf_filename": "Qwen3-1.7B-Q8_0.gguf",
    },
    "Qwen/Qwen3-8B": {
        "disable_thinking": True,
        "hf_repo": "Qwen/Qwen3-8B-GGUF",
        "hf_filename": "Qwen3-8B-Q8_0.gguf",
    },
    "Qwen/Qwen3-32B": {
        "disable_thinking": True,
        "hf_repo": "Qwen/Qwen3-32B-GGUF",
        "hf_filename": "Qwen3-32B-Q8_0.gguf",
    },
    "allenai/OLMo-2-0425-1B-Instruct": {
        "hf_repo": "allenai/OLMo-2-0425-1B-Instruct-GGUF",
        "hf_filename": "OLMo-2-0425-1B-Instruct-Q8_0.gguf",
    },
    "allenai/OLMo-2-1124-7B-Instruct": {
        "hf_repo": "bartowski/OLMo-2-1124-7B-Instruct-GGUF",
        "hf_filename": "OLMo-2-1124-7B-Instruct-Q8_0.gguf",
    },
    "allenai/OLMo-2-0325-32B-Instruct": {
        "hf_repo": "allenai/OLMo-2-0325-32B-Instruct-GGUF",
        "hf_filename": "OLMo-2-0325-32B-Instruct-Q8_0.gguf",
    },
    # Gemma 3's template merges the system message into the first user turn
    # (Gemma-2 rejected a system role), so the judge's system+user pair works.
    "google/gemma-3-1b-it": {
        "hf_repo": "bartowski/google_gemma-3-1b-it-GGUF",
        "hf_filename": "google_gemma-3-1b-it-Q8_0.gguf",
    },
    "google/gemma-3-4b-it": {
        "hf_repo": "bartowski/google_gemma-3-4b-it-GGUF",
        "hf_filename": "google_gemma-3-4b-it-Q8_0.gguf",
    },
    "google/gemma-3-12b-it": {
        "hf_repo": "bartowski/google_gemma-3-12b-it-GGUF",
        "hf_filename": "google_gemma-3-12b-it-Q8_0.gguf",
    },
    "google/gemma-3-27b-it": {
        "hf_repo": "bartowski/google_gemma-3-27b-it-GGUF",
        "hf_filename": "google_gemma-3-27b-it-Q8_0.gguf",
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "hf_repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "hf_filename": "Llama-3.2-1B-Instruct-Q8_0.gguf",
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "hf_repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "hf_filename": "Llama-3.2-3B-Instruct-Q8_0.gguf",
    },
    # InternLM2.5 — ChatML-style template with a real system role.
    "internlm/internlm2_5-1_8b-chat": {
        "hf_repo": "bartowski/internlm2_5-1_8b-chat-GGUF",
        "hf_filename": "internlm2_5-1_8b-chat-Q8_0.gguf",
    },
    "internlm/internlm2_5-7b-chat": {
        "hf_repo": "bartowski/internlm2_5-7b-chat-GGUF",
        "hf_filename": "internlm2_5-7b-chat-Q8_0.gguf",
    },
    "internlm/internlm2_5-20b-chat": {
        "hf_repo": "bartowski/internlm2_5-20b-chat-GGUF",
        "hf_filename": "internlm2_5-20b-chat-Q8_0.gguf",
    },
    # Ministral 3 (Mistral) — GGUFs from ggml-org. Mistral templates merge
    # the system message into the first [INST] block.
    "mistralai/Ministral-3-3B-Instruct-2512": {
        "hf_repo": "ggml-org/Ministral-3-3B-Instruct-2512-GGUF",
        "hf_filename": "Ministral-3-3B-Instruct-2512-Q8_0.gguf",
    },
    "mistralai/Ministral-3-8B-Instruct-2512": {
        "hf_repo": "ggml-org/Ministral-3-8B-Instruct-2512-GGUF",
        "hf_filename": "Ministral-3-8B-Instruct-2512-Q8_0.gguf",
    },
    "mistralai/Ministral-3-14B-Instruct-2512": {
        "hf_repo": "ggml-org/Ministral-3-14B-Instruct-2512-GGUF",
        "hf_filename": "Ministral-3-14B-Instruct-2512-Q8_0.gguf",
    },
}


def _local_backend() -> str:
    """Resolve the local-judge backend from env. Default: 'gguf' (Mac)."""
    return os.environ.get("JUDGE_BACKEND", "gguf").lower()


def make_judge(name: str, **overrides) -> Judge:
    """Construct the Judge for a canonical judge name.

    `overrides` forward to the concrete constructor (e.g. `client=` for
    OpenAIJudge, `gguf_path=` for LlamaCppJudge in tests). `think=True`
    requests the GGUF free-form reasoning decode and is GGUF-only; the hf
    and OpenAI paths reject it (use `--reasoning-effort` for gpt-5*).
    """
    think = overrides.pop("think", False)
    if name in _LOCAL_JUDGES:
        kwargs = {**_LOCAL_JUDGES[name], **overrides}
        backend = _local_backend()
        if backend == "gguf":
            if think:
                # Drop the grammar suppressor and widen the context window.
                kwargs["disable_thinking"] = False
                kwargs.setdefault("n_ctx", 16384)
            return LlamaCppJudge(name, think=think, **kwargs)
        if backend == "hf":
            if think:
                raise NotImplementedError(
                    "think mode is implemented only for the GGUF backend "
                    "(JUDGE_BACKEND unset). The hf path's outlines decoder is "
                    "single-pass schema-constrained; a thinking variant would "
                    "need a two-phase generator."
                )
            from core.judges.transformers_judge import TransformersJudge
            return TransformersJudge(name, **kwargs)
        raise ValueError(
            f"JUDGE_BACKEND={backend!r} not understood; "
            f"expected 'gguf' (Mac, default) or 'hf' (cluster)"
        )
    if think:
        raise ValueError(
            f"think mode requires a registered local Qwen3 judge; {name!r} "
            f"routes to the OpenAI backend, which has no GGUF thinking path. "
            f"Use --reasoning-effort for gpt-5* reasoning instead."
        )
    return OpenAIJudge(name, **overrides)


__all__ = ["Judge", "OpenAIJudge", "LlamaCppJudge", "make_judge"]
