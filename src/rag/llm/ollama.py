"""Ollama client -- local generation, the default backend.

Ollama exposes an HTTP API on localhost. Talking to it directly with httpx
avoids another dependency, and the surface used here is two endpoints.

On this machine Ollama reaches the Radeon through Vulkan/ROCm, which is why
generation runs on the GPU even though the embedding model runs on CPU
(PyTorch has no ROCm build for Windows).
"""

from __future__ import annotations

import logging

import httpx

from .base import LLMResponse

log = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct",
        timeout: int = 120,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def health(self) -> bool:
        """True only if the server responds *and* holds the requested model."""
        try:
            resp = httpx.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
        except Exception as exc:
            log.debug("Ollama unreachable: %s", exc)
            return False

        names = {m.get("name", "") for m in resp.json().get("models", [])}
        if self.model in names:
            return True
        # Ollama reports "qwen2.5:7b-instruct"; users often configure "qwen2.5".
        return any(n.split(":")[0] == self.model.split(":")[0] for n in names)

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            resp = httpx.post(
                f"{self.host}/api/generate", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Ollama request failed ({exc}). Is `ollama serve` running "
                f"and `ollama pull {self.model}` done?"
            ) from exc

        data = resp.json()
        return LLMResponse(
            text=(data.get("response") or "").strip(),
            model=self.model,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )
