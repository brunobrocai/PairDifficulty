"""LlamaCppJudge: in-process GGUF inference via llama-cpp-python.

Structured output uses llama.cpp's GBNF grammar (`response_format` with
our JSON schema), so verdicts are schema-valid by construction. The same
schema is shared with OpenAI; `_apply_strict` keeps the two backends'
schemas identical.

`disable_thinking=True` appends Qwen3's `/no_think` directive to the user
message (llama-cpp-python 0.3.x ignores `chat_template_kwargs`, so the
directive is the only way to skip the thinking block).

`think=True` instead drops the grammar and decodes free-form (`/think`,
wider `n_ctx`) so a reasoning model can emit `<think>...</think>` before
its answer; the verdict is parsed best-effort (`_parse_think_verdict`),
and ambiguous or ill-formed output becomes a null verdict rather than a
guess.

GGUF source, in priority order:
1. `gguf_path=` constructor arg.
2. `GGUF_<NAME>_PATH` env var (name uppercased, non-alphanumerics -> `_`;
   e.g. `Qwen/Qwen3-8B` -> `GGUF_QWEN_QWEN3_8B_PATH`).
3. `hf_repo` + `hf_filename` via `Llama.from_pretrained` (downloads the
   Q8_0 GGUF to the HF cache on first use; the factory registers these).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from core.judges.base import Judge
from core.judges.openai_judge import _apply_strict
from core.prompts import Prompt


def gguf_env_var(judge_name: str) -> str:
    """Convention: judge_name -> env var that holds its GGUF path."""
    sanitized = "".join(
        c if c.isalnum() else "_" for c in judge_name.upper()
    )
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return f"GGUF_{sanitized.strip('_')}_PATH"


def resolve_gguf_path(judge_name: str) -> Path | None:
    """Locally-pinned GGUF via `GGUF_<NAME>_PATH`, or None if unset.

    Raises FileNotFoundError if the env var is set but the file is
    missing (user error, distinct from "no override configured").
    """
    var = gguf_env_var(judge_name)
    raw = os.getenv(var)
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"{var}={raw!r} points at a missing file (judge {judge_name!r})"
        )
    return path


class LlamaCppJudge(Judge):
    """In-process GGUF-backed judge. One instance == one loaded model.

    Construction loads the model into VRAM/unified memory (seconds to
    tens of seconds), so reuse one judge across many .call() invocations.
    """

    def __init__(
        self,
        name: str,
        *,
        gguf_path: Path | str | None = None,
        hf_repo: str | None = None,
        hf_filename: str | None = None,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        max_tokens: int = 64,
        disable_thinking: bool = False,
        think: bool = False,
        think_max_tokens: int = 2048,
        verbose: bool = False,
    ) -> None:
        super().__init__(name)
        self.max_tokens = max_tokens
        self._disable_thinking = disable_thinking
        self._think = think
        self.think_max_tokens = think_max_tokens

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for LlamaCppJudge. "
                "Install with: uv pip install -e \".[local]\""
            ) from e

        local_path: Path | None
        if gguf_path is not None:
            local_path = Path(gguf_path).expanduser()
        else:
            local_path = resolve_gguf_path(name)

        if local_path is not None:
            self.gguf_path = local_path
            self.llm = Llama(
                model_path=str(local_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )
        elif hf_repo and hf_filename:
            self.gguf_path = None
            self.llm = Llama.from_pretrained(
                repo_id=hf_repo,
                filename=hf_filename,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )
        else:
            raise RuntimeError(
                f"No GGUF source for judge {name!r}. Either set "
                f"{gguf_env_var(name)}=/path/to/model.gguf, pass "
                f"gguf_path=..., or register hf_repo/hf_filename in "
                f"core.judges._LOCAL_GGUF."
            )

    def call(
        self,
        prompt: Prompt,
        *,
        seed: int,
        temperature: float,
        reasoning_effort: str | None,
        schema_name: str,
    ) -> tuple[dict | None, dict]:
        # Ignored for local judges: gpt-5*-style effort doesn't map onto
        # the GGUF stack. Reasoning is requested at construction (think=True).
        del reasoning_effort

        if self._think:
            return self._call_think(prompt, seed, temperature)

        schema = _apply_strict(prompt.extras["schema"])
        response_format = {"type": "json_object", "schema": schema}

        user_content = prompt.userprompt
        if self._disable_thinking:
            user_content = f"{user_content} /no_think"

        # One attempt; retries live at the orchestration layer (--retry-failed).
        try:
            raw = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": prompt.sysprompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                seed=seed,
                max_tokens=self.max_tokens,
                response_format=response_format,
            )
            return _parse(raw), raw
        except Exception as e:
            return None, {"error": f"{type(e).__name__}: {e}"}

    def _call_think(
        self,
        prompt: Prompt,
        seed: int,
        temperature: float,
    ) -> tuple[dict | None, dict]:
        """Free-form reasoning decode (no grammar), then best-effort verdict
        parse via `_parse_think_verdict`."""
        messages = [
            {"role": "system", "content": prompt.sysprompt},
            # /think is explicit even on non-thinking-default templates; a
            # no-op where thinking is already the default.
            {"role": "user", "content": f"{prompt.userprompt} /think"},
        ]
        try:
            raw = self.llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_tokens=self.think_max_tokens,
            )
        except Exception as e:
            return None, {"error": f"{type(e).__name__}: {e}"}

        text = raw["choices"][0]["message"]["content"] or ""
        # Persist the full trace (think block + answer) — the point of a think run.
        raw["reasoning"] = text
        return _parse_think_verdict(text), raw


def _parse(raw: dict) -> dict | None:
    try:
        text = raw["choices"][0]["message"]["content"]
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


_VERDICT_RE = re.compile(r"\b([AB])\b")


def _parse_think_verdict(text: str) -> dict | None:
    """Best-effort verdict from a free-form (thinking) generation.

    Keep only the answer after the last `</think>`, and accept it only if
    exactly one verdict letter (A xor B) appears. Both, neither, or a
    never-closed think block -> None (no guessing).
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    distinct = set(_VERDICT_RE.findall(text.upper()))
    if distinct == {"A"}:
        return {"verdict": "A"}
    if distinct == {"B"}:
        return {"verdict": "B"}
    return None
