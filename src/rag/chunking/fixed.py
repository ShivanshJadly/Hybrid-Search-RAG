"""Fixed-size chunking with overlap.

The baseline. Cuts on *token* windows rather than characters, because the
ceiling that actually matters is the embedding model's token limit, not string
length. Windows map back to character spans so labelling stays uniform across
strategies.

Ignores document structure entirely, which is the point: it is the control the
other two strategies are measured against.
"""

from __future__ import annotations

from ..models import LoadedDocument
from .base import Span, Tokenizer


def _snap_left(text: str, i: int) -> int:
    """Walk back to a word boundary so chunks do not begin mid-word."""
    while i > 0 and (text[i - 1].isalnum() or text[i - 1] == "'"):
        i -= 1
    return i


def _snap_right(text: str, i: int) -> int:
    n = len(text)
    while i < n and (text[i].isalnum() or text[i] == "'"):
        i += 1
    return i


class FixedSizeChunker:
    name = "fixed"

    def __init__(
        self,
        tokenizer: Tokenizer,
        chunk_tokens: int = 400,
        overlap_tokens: int = 60,
        min_chars: int = 80,
    ) -> None:
        if overlap_tokens >= chunk_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_tokens")
        self.tokenizer = tokenizer
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chars = min_chars

    def spans(self, doc: LoadedDocument) -> list[Span]:
        text = doc.text
        if not text.strip():
            return []

        offsets = self.tokenizer.offsets(text)
        if not offsets:
            return []

        # A document that fits in one window needs no slicing.
        if len(offsets) <= self.chunk_tokens:
            return [Span(0, len(text))]

        step = self.chunk_tokens - self.overlap_tokens
        out: list[Span] = []
        i = 0
        n = len(offsets)

        while i < n:
            j = min(i + self.chunk_tokens, n)
            raw_start, raw_end = offsets[i][0], offsets[j - 1][1]
            start = _snap_left(text, raw_start)
            end = _snap_right(text, raw_end)

            # Snapping completes a partial word at each edge, which with a
            # subword tokenizer adds tokens the window did not budget for --
            # observed as 402 tokens against a 400 budget. Keep the tidier
            # boundaries only when they still fit; otherwise fall back to the
            # exact token window, which is chunk_tokens by construction.
            if self.tokenizer.count(text[start:end]) > self.chunk_tokens:
                start, end = raw_start, raw_end

            # Guard against a snap producing a subset of the previous span.
            if out and start <= out[-1].start:
                start = out[-1].start + 1
            if end > start:
                out.append(Span(start, end))

            if j >= n:
                break
            i += step

        # Drop a trailing sliver, but only if the chunk before it already
        # covers the same text through its overlap.
        if len(out) > 1:
            last = out[-1]
            if len(last) < self.min_chars and last.end <= out[-2].end:
                out.pop()

        return out
