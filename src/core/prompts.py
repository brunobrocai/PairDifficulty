"""YAML-backed prompt templates with versioning + hash verification.

A prompt template is a YAML file:

    sysprompt: |
      You are an expert grader...
    userprompt: |
      Essay: {essay}
      Score it 1-100.
    extra:
      schema:
        type: object
        properties: {score: {type: integer, minimum: 1, maximum: 100}}
        required: [score]
      response_field: score
      response_range: [1, 100]

Each version is registered with a `PromptRegistry`; the run loop checks
the version's content hash against the cache manifest before any LLM call,
so editing a prompt after a cache exists is caught immediately (convention:
never edit a prompts file once a cache exists — bump the version).

Rendering is `str.format`-style substitution via `template.render(**fields)`.
Optional block composition (the `blocks` arg) resolves `{{ ns.key }}` refs
at load time, recursively, and hashes the RESOLVED text.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_BLOCK_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
_MAX_BLOCK_PASSES = 8


def _resolve_blocks(
    text: str,
    blocks: dict[str, dict[str, str]],
    source: str,
) -> str:
    """Recursively substitute `{{ ns.key }}` refs in `text` using `blocks`.

    Blocks may reference other blocks; substitution iterates until either
    a fixed point or `_MAX_BLOCK_PASSES` is hit (then raises — indicates a
    cycle or missing terminal). Missing namespace or key raises KeyError.
    """
    missing: list[str] = []

    def _repl(m: re.Match[str]) -> str:
        ns, key = m.group(1), m.group(2)
        if ns not in blocks:
            missing.append(
                f"{ns}.{key} (unknown namespace; available: {sorted(blocks)})"
            )
            return m.group(0)
        if key not in blocks[ns]:
            missing.append(
                f"{ns}.{key} (unknown key; available in {ns}: "
                f"{sorted(blocks[ns])})"
            )
            return m.group(0)
        return blocks[ns][key]

    for _ in range(_MAX_BLOCK_PASSES):
        missing.clear()
        new_text = _BLOCK_RE.sub(_repl, text)
        if missing:
            raise KeyError(
                f"{source}: unresolved block refs: {missing}"
            )
        if new_text == text:
            return text
        text = new_text
    raise RecursionError(
        f"{source}: block substitution did not converge after "
        f"{_MAX_BLOCK_PASSES} passes (cycle or terminal block missing)"
    )


@dataclass(frozen=True)
class Prompt:
    """A rendered prompt: ready-to-send sysprompt + userprompt + extras."""
    sysprompt: str
    userprompt: str
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptTemplate:
    """An unrendered template: file contents + identity + content hash."""

    key: str
    version: str
    path: Path
    sysprompt: str
    userprompt: str
    extras: dict[str, Any]
    content_hash: str

    @classmethod
    def from_yaml(
        cls,
        key: str,
        version: str,
        path: Path,
        blocks: dict[str, dict[str, str]] | None = None,
    ) -> "PromptTemplate":
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top level must be a mapping")
        sysprompt = data.get("sysprompt", "")
        userprompt = data.get("userprompt", "")
        extras = data.get("extra", {}) or {}
        if not isinstance(sysprompt, str) or not isinstance(userprompt, str):
            raise ValueError(f"{path}: sysprompt and userprompt must be strings")
        if not isinstance(extras, dict):
            raise ValueError(f"{path}: 'extra' must be a mapping")
        if blocks:
            sysprompt = _resolve_blocks(sysprompt, blocks, f"{path} sysprompt")
            userprompt = _resolve_blocks(userprompt, blocks, f"{path} userprompt")
        # Hash the semantic content (resolved), not the file text, so
        # cosmetic edits don't trip the drift check but block changes do.
        payload = {
            "sysprompt": sysprompt,
            "userprompt": userprompt,
            "extra": extras,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        h = hashlib.sha256(blob).hexdigest()
        return cls(
            key=key, version=version, path=path,
            sysprompt=sysprompt, userprompt=userprompt,
            extras=extras, content_hash=h,
        )

    def required_fields(self) -> set[str]:
        """Names of every {placeholder} that appears in sys+user prompt."""
        return set(_PLACEHOLDER_RE.findall(self.sysprompt)) | set(
            _PLACEHOLDER_RE.findall(self.userprompt)
        )

    def render(self, **fields: Any) -> Prompt:
        """Substitute every {placeholder} with the matching kwarg.

        Raises KeyError listing every unbound placeholder, instead of
        leaving stray `{essay}` in the rendered prompt.
        """
        needed = self.required_fields()
        missing = needed - set(fields)
        if missing:
            raise KeyError(
                f"prompt {self.key}@{self.version} missing fields: "
                f"{sorted(missing)}; got {sorted(fields)}"
            )
        rendered_sys = _PLACEHOLDER_RE.sub(
            lambda m: str(fields[m.group(1)]), self.sysprompt
        )
        rendered_user = _PLACEHOLDER_RE.sub(
            lambda m: str(fields[m.group(1)]), self.userprompt
        )
        return Prompt(
            sysprompt=rendered_sys,
            userprompt=rendered_user,
            extras=dict(self.extras),
        )


class PromptRegistry:
    """Global registry: (key, version) -> PromptTemplate."""

    def __init__(self):
        self._by_key: dict[str, dict[str, PromptTemplate]] = {}

    def register(
        self,
        key: str,
        version: str,
        path: Path | str,
        blocks: dict[str, dict[str, str]] | None = None,
    ) -> PromptTemplate:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"prompt file not found: {path}")
        tpl = PromptTemplate.from_yaml(key, version, path, blocks=blocks)
        self._by_key.setdefault(key, {})[version] = tpl
        return tpl

    def register_dir(
        self,
        key: str,
        directory: Path | str,
        blocks: dict[str, dict[str, str]] | None = None,
    ) -> list[PromptTemplate]:
        """Auto-register every v*.yaml in `directory` (e.g. v1.yaml -> 'v1')."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"prompt dir not found: {directory}")
        registered: list[PromptTemplate] = []
        for p in sorted(directory.glob("v*.yaml")):
            version = p.stem  # 'v1', 'v2', ...
            registered.append(self.register(key, version, p, blocks=blocks))
        if not registered:
            raise FileNotFoundError(
                f"no v*.yaml files in {directory}"
            )
        return registered

    def get(self, key: str, version: str) -> PromptTemplate:
        versions = self._by_key.get(key)
        if not versions:
            raise KeyError(
                f"no prompt registered under {key!r} "
                f"(known keys: {sorted(self._by_key)})"
            )
        tpl = versions.get(version)
        if tpl is None:
            raise KeyError(
                f"prompt {key!r} has no version {version!r} "
                f"(known versions: {sorted(versions)})"
            )
        return tpl

    def list_versions(self, key: str) -> list[str]:
        return sorted(self._by_key.get(key, {}))

    def list_keys(self) -> list[str]:
        return sorted(self._by_key)


# Module-level default registry — pilots can use this directly, or
# instantiate their own for tests.
REGISTRY = PromptRegistry()


def register(
    key: str,
    version: str,
    path: Path | str,
    blocks: dict[str, dict[str, str]] | None = None,
) -> PromptTemplate:
    return REGISTRY.register(key, version, path, blocks=blocks)


def register_dir(
    key: str,
    directory: Path | str,
    blocks: dict[str, dict[str, str]] | None = None,
) -> list[PromptTemplate]:
    return REGISTRY.register_dir(key, directory, blocks=blocks)


def get(key: str, version: str) -> PromptTemplate:
    return REGISTRY.get(key, version)


def load_blocks_file(path: Path | str) -> dict[str, str]:
    """Load a flat `{name: text}` YAML file used as a single-namespace
    block source. Used by dataset prompts.py modules to assemble the
    namespace dict passed to register/register_dir.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            raise ValueError(f"{path}: block key must be a string, got {k!r}")
        if not isinstance(v, str):
            raise ValueError(
                f"{path}: block {k!r} must be a string, got {type(v).__name__}"
            )
        out[k] = v
    return out
