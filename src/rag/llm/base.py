"""The LLM seam.

Generation, citation verification and eval judging all go through this one
interface, so swapping a local model for a free-tier API is a config change
rather than a rewrite. Implementations: OllamaClient (local), GroqClient and
GeminiClient (free tier), StubLLM (offline tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient(Protocol):
    """Every LLM call in this project goes through this."""

    model: str

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...

    def health(self) -> bool:
        """Cheap reachability check, so a misconfigured backend fails at
        startup rather than midway through a 150-question eval sweep."""
        ...
