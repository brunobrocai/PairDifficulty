"""OpenAI client factory: load .env and instantiate `openai.OpenAI`.

Single entry point so the runners don't reimplement the dotenv + key-lookup
dance. Defaults to the standard `OPENAI_API_KEY` env var (set it in a `.env`
at the package root to regenerate caches against the live API).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dotenv import find_dotenv, load_dotenv

if TYPE_CHECKING:
    from openai import OpenAI


def get_openai_client(
    env_var: str = "OPENAI_API_KEY",
    *,
    org_env_var: str | None = None,
) -> "OpenAI":
    """Load .env (searching upward from cwd) and return an OpenAI client."""
    from openai import OpenAI

    dotenv_path = find_dotenv(usecwd=True)
    if not dotenv_path:
        raise FileNotFoundError(
            "no .env file found searching upward from cwd; cannot resolve API key"
        )
    load_dotenv(dotenv_path)
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"env var {env_var!r} not set in {dotenv_path}")
    org = os.getenv(org_env_var) if org_env_var else None
    return OpenAI(api_key=api_key, organization=org)
