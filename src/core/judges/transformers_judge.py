"""TransformersJudge: HuggingFace transformers + outlines, in BF16.

Loads `AutoModelForCausalLM` in BF16 with `device_map="auto"`. The JSON
schema is compiled with `outlines` to constrain decoding, so the output
always matches the schema.
"""

from __future__ import annotations

import hashlib
import json
import signal
import sys
from typing import Any

from core.judges.base import Judge
from core.judges.openai_judge import _apply_strict
from core.prompts import Prompt


class TransformersJudge(Judge):
    """HF transformers judge with outlines-grammar-constrained JSON.

    One instance owns one loaded model; construction is dominated by
    `from_pretrained` (tens of seconds for 32B BF16 warm). Reuse it.
    """

    def __init__(
        self,
        name: str,
        *,
        disable_thinking: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
        max_new_tokens: int = 1024,
        torch_dtype: str = "bfloat16",
        # Unused here; accepted so both local backends take the same kwargs.
        hf_repo: str | None = None,
        hf_filename: str | None = None,
    ) -> None:
        super().__init__(name)
        del hf_repo, hf_filename

        # disable_thinking maps to enable_thinking=False.
        ctk = dict(chat_template_kwargs or {})
        if disable_thinking:
            ctk.setdefault("enable_thinking", False)
        self._chat_template_kwargs = ctk
        self.max_new_tokens = max_new_tokens

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers + torch are required for TransformersJudge. "
                "Install with: pip install -e \".[cluster]\""
            ) from e

        self._torch = torch
        dtype = getattr(torch, torch_dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self._hf_model.eval()

        _gpu_preflight(self._hf_model)

        try:
            import outlines
            from outlines.types import JsonSchema
        except ImportError as e:
            raise ImportError(
                "outlines is required for TransformersJudge schema enforcement. "
                "Install with: pip install -e \".[cluster]\""
            ) from e

        self._outlines = outlines
        self._json_schema_cls = JsonSchema
        # outlines 1.x: wrap the already-loaded HF model + tokenizer.
        self._model = outlines.from_transformers(self._hf_model, self._tokenizer)
        self._generators: dict[str, Any] = {}

    def _get_generator(self, schema: dict) -> Any:
        """Compile (and memoize) an outlines Generator for this JSON schema.

        Cached on a hash of the schema, because compiling takes seconds.
        """
        key = hashlib.md5(
            json.dumps(schema, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if key not in self._generators:
            self._generators[key] = self._outlines.Generator(
                self._model, self._json_schema_cls(schema),
            )
        return self._generators[key]

    def call(
        self,
        prompt: Prompt,
        *,
        seed: int,
        temperature: float,
        reasoning_effort: str | None,
        schema_name: str,
    ) -> tuple[dict | None, dict]:
        # reasoning_effort is ignored: none of our HF judges are reasoning
        # models; Qwen3 thinking is off via chat_template_kwargs.
        del reasoning_effort

        if "schema" not in prompt.extras:
            raise ValueError(
                "prompt is missing extras['schema']; structured output requires it"
            )
        schema = _apply_strict(prompt.extras["schema"])

        text = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": prompt.sysprompt},
                {"role": "user", "content": prompt.userprompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            **self._chat_template_kwargs,
        )

        generator = self._get_generator(schema)

        try:
            # Greedy at temperature 0; the seed only matters for sampling.
            self._torch.manual_seed(seed)
            out = generator(text, max_new_tokens=self.max_new_tokens)
            # outlines 1.x returns the raw, schema-constrained JSON string.
            parsed = json.loads(out)
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"outlines returned non-object JSON: {type(parsed).__name__}"
                )
            raw = {
                "backend": "transformers",
                "model": self.name,
                "choices": [{"message": {"content": out}}],
            }
            return parsed, raw
        except Exception as e:
            return None, {
                "error": f"{type(e).__name__}: {e}",
                "backend": "transformers",
            }


def _gpu_preflight(model) -> None:
    """Fail fast if HF spilled layers to CPU.

    A model that offloads even a few layers runs far slower.
    """
    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        print("[TransformersJudge] WARNING: no CUDA — running on CPU",
              file=sys.stderr)
        return

    free, total = torch.cuda.mem_get_info()
    print(
        f"[TransformersJudge] GPU memory: "
        f"{free / 1e9:.1f} / {total / 1e9:.1f} GB free",
        file=sys.stderr,
    )

    device_map = getattr(model, "hf_device_map", None)
    if device_map is None:
        return

    cpu_layers = [k for k, v in device_map.items() if str(v) == "cpu"]
    if cpu_layers:
        print(
            f"[TransformersJudge] FATAL: {len(cpu_layers)} layers on CPU "
            f"(first 5: {cpu_layers[:5]}). Aborting — request a node "
            f"with more GPU memory.",
            file=sys.stderr,
        )
        # SIGTERM rather than sys.exit so the surrounding sbatch script
        # sees a non-zero exit and the SLURM job is marked FAILED.
        signal.raise_signal(signal.SIGTERM)
        sys.exit(1)
