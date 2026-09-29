"""Judge ABC: the thing a BaseRun talks to instead of a raw API client.

Each subclass hides one backend (OpenAI, llama.cpp, ...) behind a single
`call(...)`; the run loop never inspects how it works.

Contract — every Judge subclass MUST:
- Set `self.name` to the canonical judge identifier (matches `cfg.judge`
  and the cache filename; the string the paper reports).
- Implement `call(prompt, *, seed, temperature, reasoning_effort,
  schema_name) -> (parsed_dict | None, raw_dict)`. One attempt; returns
  `(None, error_dict)` on any exception or parse failure. Retries live
  at the orchestration layer (`--retry-failed`), not inside the judge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.prompts import Prompt


class Judge(ABC):
    """Abstract judge model. Subclasses encapsulate one backend.

    Subclasses store whatever state the backend needs (an OpenAI client,
    a loaded llama.cpp model, ...) on `self`. The run loop never inspects
    that state — it only calls `.call(...)`.
    """

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("Judge.name must be a non-empty string")
        self.name = name

    @abstractmethod
    def call(
        self,
        prompt: Prompt,
        *,
        seed: int,
        temperature: float,
        reasoning_effort: str | None,
        schema_name: str,
    ) -> tuple[dict | None, dict]:
        """Issue one LLM call and return (parsed_obj_or_None, raw_response).

        - `parsed`: the structured-output JSON object, or None if the
          single attempt failed (malformed JSON, schema-rejection,
          transport error). Caller treats None as a failure row.
        - `raw`: the full backend response dict (or a stand-in dict on
          transport-error). Stored on the cache row for diagnostics.
        """
        ...
