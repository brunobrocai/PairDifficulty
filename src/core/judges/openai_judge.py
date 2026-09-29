"""OpenAIJudge: Chat Completions + Responses API, routed by model name.

Routing:
- `gpt-5*`, `o1*`, `o3*` -> Responses API (these are the reasoning-
  capable families; the Responses API is OpenAI's intended surface for
  them even when `reasoning_effort="none"`).
- everything else -> Chat Completions with structured outputs.

Strict-mode JSON schema enforcement is handled by recursively setting
`additionalProperties: False` on every object schema before wrapping it
in OpenAI's response_format envelope. We deep-copy so we never mutate
the registry's stored schema.

Schema-name sanitization: OpenAI rejects names with `.` (which our
prompt keys contain, e.g. `clear.pairwise`), so we strip every
non-alphanumeric character to `_` before sending.
"""

from __future__ import annotations

import copy
import json
from typing import Any, TYPE_CHECKING

from core.judges.base import Judge
from core.prompts import Prompt

if TYPE_CHECKING:
    from openai import OpenAI


def _is_reasoning_model(model: str) -> bool:
    return model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")


def _apply_strict(schema: dict) -> dict:
    """Recursively set additionalProperties=False on every object schema."""
    out = copy.deepcopy(schema)
    _apply_strict_in_place(out)
    return out


def _apply_strict_in_place(schema: dict) -> None:
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
    for v in (schema.get("properties") or {}).values():
        _apply_strict_in_place(v)
    for v in (schema.get("$defs") or {}).values():
        _apply_strict_in_place(v)
    if "items" in schema:
        _apply_strict_in_place(schema["items"])


def _schema_for_chat(name: str, schema: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": _apply_strict(schema),
            "strict": True,
        },
    }


def _schema_for_responses(name: str, schema: dict) -> dict:
    return {
        "type": "json_schema",
        "name": name,
        "schema": _apply_strict(schema),
        "strict": True,
    }


def _extract_schema_and_name(prompt: Prompt, default_name: str) -> tuple[dict, str]:
    if "schema" not in prompt.extras:
        raise ValueError(
            "prompt is missing extras['schema']; structured output requires it"
        )
    schema = prompt.extras["schema"]
    name = prompt.extras.get("schema_name") or default_name
    name = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    return schema, name


class OpenAIJudge(Judge):
    """OpenAI-backed judge. Owns its OpenAI client.

    The client is lazily constructed from the project's `.env` (via
    `core.llm_client.get_openai_client`) unless one is injected by the
    caller — injection is useful for tests and for sharing a client
    across many runs in the same process.
    """

    def __init__(self, name: str, *, client: "OpenAI | None" = None) -> None:
        super().__init__(name)
        if client is None:
            from core.llm_client import get_openai_client
            client = get_openai_client()
        self.client = client
        self._use_responses = _is_reasoning_model(name)

    def call(
        self,
        prompt: Prompt,
        *,
        seed: int,
        temperature: float,
        reasoning_effort: str | None,
        schema_name: str,
    ) -> tuple[dict | None, dict]:
        try:
            if self._use_responses:
                raw = self._call_responses(
                    prompt, reasoning_effort, schema_name,
                )
                parsed = _parse_responses(raw)
            else:
                raw = self._call_chat(
                    prompt, temperature, seed, schema_name,
                )
                parsed = _parse_chat(raw)
            return parsed, raw
        except Exception as e:
            return None, {"error": f"{type(e).__name__}: {e}"}

    def _call_chat(
        self, prompt: Prompt, temperature: float, seed: int, schema_name: str,
    ) -> dict:
        schema, name = _extract_schema_and_name(prompt, schema_name)
        completion = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "system", "content": prompt.sysprompt},
                {"role": "user", "content": prompt.userprompt},
            ],
            temperature=temperature,
            seed=seed,
            response_format=_schema_for_chat(name, schema),
        )
        return completion.to_dict()

    def _call_responses(
        self, prompt: Prompt, reasoning_effort: str | None, schema_name: str,
    ) -> dict:
        schema, name = _extract_schema_and_name(prompt, schema_name)
        extra: dict[str, Any] = {}
        if reasoning_effort and reasoning_effort != "none":
            extra["reasoning"] = {"effort": reasoning_effort}
        completion = self.client.responses.create(
            instructions=prompt.sysprompt,
            input=prompt.userprompt,
            text={"format": _schema_for_responses(name, schema)},
            model=self.name,
            **extra,
        )
        return completion.to_dict()


def _parse_chat(raw: dict) -> dict | None:
    try:
        text = raw["choices"][0]["message"]["content"]
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _parse_responses(raw: dict) -> dict | None:
    try:
        text = raw["output"][-1]["content"][-1]["text"]
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
