"""Free-tier API clients, used for judging (Notion plan, Appendix A.4).

The plan's split: a local model generates, a free-tier API grades. Grading is
where a small local model is weakest, and grading is low-volume enough to fit
inside a free tier.

Both clients handle HTTP 429 with backoff, because both providers return it on
quota exhaustion and an unhandled 429 halfway through a 150-question sweep
wastes the day's quota. Groq's binding limit is tokens-per-minute, not
requests -- a judge call carrying retrieved context runs 2-3K tokens against
an 8K/min ceiling, so roughly three calls per minute.
"""

from __future__ import annotations

import logging
import random
import re
import time

import httpx

from .base import LLMResponse

log = logging.getLogger(__name__)


class RateLimited(RuntimeError):
    pass


def _daily_quota_exhausted(resp: httpx.Response) -> str:
    """The model name whose *per-day* free quota is spent, or "".

    A 429 is worth waiting out when the limit is per-minute and worthless
    when it is per-day: Google still sends a RetryInfo of half a minute for
    a daily cap, so backoff burns five sleeps and fails anyway. Preview
    models are where this bites -- some carry a free-tier allowance of a few
    dozen requests per day, far under one evaluation sweep.
    """
    try:
        details = resp.json().get("error", {}).get("details", [])
    except Exception:
        return ""
    for detail in details:
        if not detail.get("@type", "").endswith("QuotaFailure"):
            continue
        for violation in detail.get("violations", []):
            if "PerDay" in violation.get("quotaId", ""):
                dims = violation.get("quotaDimensions", {})
                return dims.get("model") or "this model"
    return ""


def _retry_delay(resp: httpx.Response) -> float | None:
    """How long the provider says to wait, in seconds, or None.

    Groq uses the standard `Retry-After` header. Google does not: it returns
    the wait inside the error body, as a RetryInfo detail and again in prose.
    Guessing instead -- doubling from two seconds -- gives up after thirty,
    which is short of the minute a per-minute quota actually needs, so a
    sweep dies on a limit that would have cleared on its own.
    """
    header = resp.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    try:
        err = resp.json().get("error", {})
    except Exception:
        return None
    for detail in err.get("details", []):
        if detail.get("@type", "").endswith("RetryInfo"):
            m = re.match(r"([\d.]+)s?$", str(detail.get("retryDelay", "")))
            if m:
                return float(m.group(1))
    m = re.search(r"retry in ([\d.]+)s", str(err.get("message", "")))
    return float(m.group(1)) if m else None


def _post_with_backoff(
    url: str,
    *,
    json: dict,
    headers: dict,
    timeout: int,
    max_attempts: int = 5,
) -> dict:
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            resp = httpx.post(url, json=json, headers=headers, timeout=timeout)
        except httpx.HTTPError as exc:
            if attempt == max_attempts:
                raise RuntimeError(f"request failed after {attempt} attempts: {exc}")
            time.sleep(delay)
            delay *= 2
            continue

        if resp.status_code == 429:
            spent = _daily_quota_exhausted(resp)
            if spent:
                raise RateLimited(
                    f"daily free-tier quota for {spent} is spent; waiting will "
                    f"not clear it before tomorrow. Point RAG_GEMINI_MODEL "
                    f"(or RAG_JUDGE_MODEL) at a model with a larger free "
                    f"allowance, or resume the run tomorrow."
                )
            if attempt == max_attempts:
                raise RateLimited(
                    f"quota exhausted after {attempt} attempts. Free-tier limits "
                    f"reset on a schedule; resume the run later."
                )
            # Honour the provider's own stated delay when it gives one, else
            # exponential backoff. Jitter either way so parallel workers do
            # not retry in lockstep.
            wait = _retry_delay(resp) or delay
            log.warning("Rate limited; sleeping %.1fs (attempt %d)", wait, attempt)
            time.sleep(wait + random.uniform(0, 1))
            delay *= 2
            continue

        resp.raise_for_status()
        return resp.json()

    raise RateLimited("exhausted retries")


class GroqClient:
    """OpenAI-compatible chat completions endpoint."""

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self, api_key: str, model: str = "llama-3.3-70b-versatile", timeout: int = 120
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def health(self) -> bool:
        return bool(self.api_key)

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        data = _post_with_backoff(
            self.URL,
            json={
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        usage = data.get("usage", {})
        return LLMResponse(
            text=data["choices"][0]["message"]["content"].strip(),
            model=self.model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )


class GeminiClient:
    """Google Generative Language API.

    Per-model free-tier limits are account-specific and shown in AI Studio
    (aistudio.google.com/rate-limit) rather than published in the docs. Check
    your own account before designing a sweep around a number.
    """

    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    # Gemini 3 spends reasoning tokens out of the same maxOutputTokens budget
    # the answer draws on, so a caller asking for 16 tokens can get an empty
    # reply with finishReason=MAX_TOKENS and never see a word. Reserve budget
    # for thinking on top of what the caller asked for, and hold thinking to
    # its low setting -- neither generation nor judging here benefits from a
    # long deliberation, and both are latency-sensitive in a 450-call sweep.
    THINKING_RESERVE = 1024

    def __init__(
        self, api_key: str, model: str = "gemini-2.0-flash", timeout: int = 120
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def health(self) -> bool:
        """A key alone is not health -- an invalid key or a retired model both
        look fine until the first real call. Ask the API instead."""
        if not self.api_key:
            return False
        try:
            names = self.list_models()
        except Exception:
            return False
        if not names:
            return False
        return any(n == self.model or n.startswith(self.model) for n in names)

    def list_models(self) -> list[str]:
        """Models this key can actually reach, bare ids without the
        `models/` prefix. Free-tier availability is account-specific and not
        published, so this is the only reliable way to know."""
        resp = httpx.get(f"{self.BASE}?key={self.api_key}", timeout=15)
        resp.raise_for_status()
        out = []
        for m in resp.json().get("models", []):
            methods = m.get("supportedGenerationMethods") or []
            if methods and "generateContent" not in methods:
                continue
            out.append(m.get("name", "").removeprefix("models/"))
        return sorted(n for n in out if n)

    def _thinks(self) -> bool:
        """thinkingConfig is a Gemini 3 field; 2.x rejects it with a 400."""
        return self.model.startswith("gemini-3") or self.model.endswith("-latest")

    def _generation_config(self, temperature: float, max_tokens: int) -> dict:
        cfg: dict = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if self._thinks():
            cfg["maxOutputTokens"] = max_tokens + self.THINKING_RESERVE
            cfg["thinkingConfig"] = {"thinkingLevel": "low"}
        return cfg

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        data = _post_with_backoff(
            f"{self.BASE}/{self.model}:generateContent?key={self.api_key}",
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": self._generation_config(temperature, max_tokens),
            },
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        candidates = data.get("candidates") or []
        if not candidates:
            # Safety filters and recitation blocks return 200 with no candidate.
            return LLMResponse(text="", model=self.model)
        parts = candidates[0].get("content", {}).get("parts", [])
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text="".join(p.get("text", "") for p in parts).strip(),
            model=self.model,
            prompt_tokens=usage.get("promptTokenCount", 0),
            completion_tokens=usage.get("candidatesTokenCount", 0),
        )
