"""Semantic chunking: cut where the topic actually changes.

Sentences are embedded, and a boundary is placed wherever the distance between
consecutive sentences jumps past a percentile threshold. The idea is that a
topic shift shows up as a spike in embedding distance, so cuts land between
ideas rather than in the middle of one.

Two honest caveats, both worth stating before Phase 4 compares this against the
simpler strategies:

* It is far more expensive than the others -- every sentence is embedded, not
  just every chunk. On a large corpus that is the dominant ingest cost.
* The percentile threshold is relative to each document. A document that is
  uniformly about one thing still gets cut at its 95th-percentile distances,
  because a percentile always finds something. This is inherent to the method,
  not a bug, and is exactly the kind of thing the evaluation should surface.

Each sentence is embedded together with its neighbours (a buffer window), which
substantially reduces noise from short sentences that carry little signal alone.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..models import LoadedDocument
from .base import Span, Tokenizer
from .splitting import sentence_spans, split_to_budget

EmbedFn = Callable[[list[str]], np.ndarray]


class SemanticChunker:
    name = "semantic"

    def __init__(
        self,
        tokenizer: Tokenizer,
        embed_fn: EmbedFn,
        chunk_tokens: int = 400,
        percentile: float = 95.0,
        buffer_size: int = 1,
        min_sentences: int = 4,
    ) -> None:
        self.tokenizer = tokenizer
        self.embed_fn = embed_fn
        self.chunk_tokens = chunk_tokens
        self.percentile = percentile
        self.buffer_size = buffer_size
        self.min_sentences = min_sentences

    def spans(self, doc: LoadedDocument) -> list[Span]:
        text = doc.text
        if not text.strip():
            return []

        sentences = sentence_spans(text)
        # Too few sentences for a percentile to mean anything.
        if len(sentences) < self.min_sentences:
            return split_to_budget(text, 0, len(text), self.tokenizer, self.chunk_tokens)

        windows = self._windows(text, sentences)
        vectors = self.embed_fn(windows)
        distances = self._distances(vectors)
        if distances.size == 0:
            return split_to_budget(text, 0, len(text), self.tokenizer, self.chunk_tokens)

        threshold = float(np.percentile(distances, self.percentile))
        breaks = [i for i, d in enumerate(distances) if d >= threshold]

        groups = self._groups(sentences, breaks)

        out: list[Span] = []
        for group in groups:
            # A semantic group can still exceed the embedding token ceiling;
            # the shared splitter enforces it the same way as everywhere else.
            out.extend(
                split_to_budget(
                    text, group.start, group.end, self.tokenizer, self.chunk_tokens
                )
            )
        return out

    def _windows(self, text: str, sentences: list[Span]) -> list[str]:
        """Each sentence plus its neighbours, so short sentences carry context."""
        out: list[str] = []
        n = len(sentences)
        for i in range(n):
            lo = max(0, i - self.buffer_size)
            hi = min(n, i + self.buffer_size + 1)
            out.append(text[sentences[lo].start : sentences[hi - 1].end].strip())
        return out

    @staticmethod
    def _distances(vectors: np.ndarray) -> np.ndarray:
        """Cosine distance between consecutive windows.

        Vectors arrive L2-normalised from the embedder, so a dot product is
        already the cosine similarity. Normalising again defensively costs
        almost nothing and stops a silent error if that ever changes.
        """
        if vectors.ndim != 2 or len(vectors) < 2:
            return np.array([])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = vectors / norms
        sims = np.sum(unit[:-1] * unit[1:], axis=1)
        return 1.0 - sims

    @staticmethod
    def _groups(sentences: list[Span], breaks: list[int]) -> list[Span]:
        """Turn breakpoint indices into contiguous sentence groups.

        A break at index i means: cut *after* sentence i.
        """
        groups: list[Span] = []
        start_idx = 0
        for b in breaks:
            if b < start_idx:
                continue
            groups.append(Span(sentences[start_idx].start, sentences[b].end))
            start_idx = b + 1
        if start_idx < len(sentences):
            groups.append(Span(sentences[start_idx].start, sentences[-1].end))
        return groups
