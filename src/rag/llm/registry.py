"""Backend selection.

Everything that calls an LLM asks for one here, so switching between a local
model, a free-tier API and the optional paid final eval is a config change --
which is the entire point of the seam described in the Notion plan, A.6.

API keys are read from the environment (`GROQ_API_KEY`, `GEMINI_API_KEY`)
rather than committed to config.
"""

from __future__ import annotations

import os

from ..config import Settings
from .base import LLMClient
from .free_api import GeminiClient, GroqClient
from .ollama import OllamaClient
from .stub import StubLLM

BACKENDS = ("ollama", "groq", "gemini", "stub")


class UnknownBackend(Exception):
    pass


def build_llm(settings: Settings, backend: str | None = None, **kwargs) -> LLMClient:
    name = (backend or settings.llm_backend).lower()

    if name == "ollama":
        return OllamaClient(
            host=settings.ollama_host,
            model=settings.ollama_model,
            timeout=settings.llm_timeout_seconds,
        )
    if name == "groq":
        return GroqClient(
            api_key=settings.groq_api_key or os.environ.get("GROQ_API_KEY", ""),
            model=settings.groq_model,
            timeout=settings.llm_timeout_seconds,
        )
    if name == "gemini":
        return GeminiClient(
            api_key=settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", ""),
            model=settings.gemini_model,
            timeout=settings.llm_timeout_seconds,
        )
    if name == "stub":
        return StubLLM(**kwargs)

    raise UnknownBackend(f"Unknown backend '{name}'. Available: {', '.join(BACKENDS)}")


def build_judge(settings: Settings, backend: str | None = None) -> LLMClient:
    """The judge is deliberately separate from the generator.

    The plan's free-tier arrangement is a local model generating and a
    stronger free-tier model grading; that only works if the two are
    configured independently.
    """
    name = backend or settings.judge_backend
    client = build_llm(settings, name, mode="judge")
    if settings.judge_model and hasattr(client, "model"):
        client.model = settings.judge_model
    return client


def describe_backend(client: LLMClient) -> str:
    return f"{type(client).__name__}({getattr(client, 'model', '?')})"
