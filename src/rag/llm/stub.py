"""Deterministic LLM stand-ins.

Real generation needs Ollama running or a free-tier API key. Neither should be
required to run the test suite, and neither should be required to prove the
answer pipeline is wired correctly -- so every stage that calls an LLM can be
driven by one of these instead.

Three modes:

* `extractive` -- pulls sentences straight out of the supplied context and
  cites them. Not intelligent, but it produces *structurally real* output:
  grounded text with `[1]`-style markers, which is enough to exercise citation
  parsing, verification and confidence scoring end to end.
* `judge` -- returns a fixed verdict, for testing the verification plumbing.
* `scripted` -- returns queued responses in order, for tests that need exact
  control.
"""

from __future__ import annotations

import re

from .base import LLMResponse

_CONTEXT_BLOCK = re.compile(r"^\[(\d+)\]\s.*?$\n(.*?)(?=\n\[\d+\]\s|\Z)", re.M | re.S)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[])")


class StubLLM:
    def __init__(
        self,
        mode: str = "extractive",
        responses: list[str] | None = None,
        judge_verdict: str = "SUPPORTED",
        max_sentences: int = 2,
    ) -> None:
        if mode not in {"extractive", "judge", "scripted"}:
            raise ValueError(f"unknown stub mode: {mode}")
        self.mode = mode
        self.model = f"stub:{mode}"
        self.responses = list(responses or [])
        self.judge_verdict = judge_verdict
        self.max_sentences = max_sentences
        self.calls: list[tuple[str, str]] = []  # inspected by tests

    def health(self) -> bool:
        return True

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls.append((system, prompt))

        if self.mode == "scripted":
            text = self.responses.pop(0) if self.responses else ""
        elif self.mode == "judge":
            text = self.judge_verdict
        else:
            text = self._extract(prompt)

        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
        )

    def _extract(self, prompt: str) -> str:
        """Answer using the first context block, cited as [1]."""
        blocks = _CONTEXT_BLOCK.findall(prompt)
        if not blocks:
            return "The provided context does not contain enough information."

        _, body = blocks[0]
        sentences = [s.strip() for s in _SENTENCE.split(body.strip()) if s.strip()]
        if not sentences:
            return "The provided context does not contain enough information."

        answer = " ".join(sentences[: self.max_sentences])
        return f"{answer} [1]"
