"""Cross-encoder reranking.

The retrievers score a query and a chunk *separately* and compare the results.
A cross-encoder reads both together in one pass, which lets it judge actual
relevance rather than vector proximity. It is far more accurate and far too
slow to run over a whole corpus -- hence the two-stage shape: retrieve 20
cheaply, rerank those 20 properly, keep 5.

Local and free (`ms-marco-MiniLM-L-6-v2`, ~80MB). The Notion plan offers
"cross-encoder / LLM-as-judge"; the cross-encoder is the right pick here
because it costs nothing per query and the judge budget is better spent on
citation verification, where nothing cheaper exists.
"""

from __future__ import annotations

import logging
from typing import Protocol

from .config import Settings
from .retrieval import RetrievedChunk

log = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        ...


class NoOpReranker:
    """Keeps fusion order. The control condition for measuring rerank lift."""

    name = "none"

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return chunks


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("Loading reranker %s", self.settings.reranker_model)
            self._model = CrossEncoder(self.settings.reranker_model, device="cpu")
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        try:
            scores = self.model.predict(
                [(query, c.text) for c in chunks], show_progress_bar=False
            )
        except Exception as exc:  # missing model, no disk, offline
            log.warning("Reranker unavailable (%s); keeping fusion order", exc)
            return chunks
        for chunk, score in zip(chunks, scores):
            # Raw logits, not probabilities: useful for ordering, meaningless
            # as an absolute number. Squashed in confidence.py where a bounded
            # value is actually needed.
            chunk.rerank_score = float(score)
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)


def build_reranker(settings: Settings) -> Reranker:
    return CrossEncoderReranker(settings) if settings.rerank_enabled else NoOpReranker()
