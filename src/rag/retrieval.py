"""Hybrid retrieval: dense + sparse, fused with Reciprocal Rank Fusion.

Why fuse on *rank* rather than score: the two retrievers produce numbers that
mean different things. Cosine similarity is bounded roughly 0..1; BM25 is
unbounded and scales with corpus statistics. Normalising them onto a common
scale requires assumptions that break whenever the corpus changes. Ranks are
directly comparable by construction, which is the whole appeal of RRF.

The formula, per document d:

    score(d) = sum over lists L of  weight(L) / (k + rank(d in L))

Documents missing from a list simply contribute nothing from it. k damps the
advantage of the very top ranks; 60 is the value from the original paper.

Every result carries where it came from -- dense rank, sparse rank, both -- so
the dashboard can show why something surfaced, and so "hybrid actually beat
dense-only" is a claim backed by data rather than an assertion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import Settings
from .embeddings import LocalEmbedder
from .stores.bm25 import Bm25Store
from .stores.vector import VectorStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    meta: dict

    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    sparse_score: Optional[float] = None

    rrf_score: float = 0.0
    rerank_score: Optional[float] = None

    @property
    def found_by(self) -> str:
        if self.dense_rank is not None and self.sparse_rank is not None:
            return "both"
        if self.dense_rank is not None:
            return "dense"
        if self.sparse_rank is not None:
            return "sparse"
        return "none"

    @property
    def final_score(self) -> float:
        """The reranker's opinion wins when it has one."""
        return self.rerank_score if self.rerank_score is not None else self.rrf_score

    def citation(self) -> str:
        cite = self.meta.get("source_name", "?")
        if self.meta.get("heading_path"):
            cite += f" > {self.meta['heading_path']}"
        if self.meta.get("page"):
            cite += f" > p.{self.meta['page']}"
        return cite


@dataclass
class RetrievalTrace:
    """What happened during one retrieval, for the dashboard and for evals."""

    query: str
    strategy: str
    dense_hits: int = 0
    sparse_hits: int = 0
    fused_candidates: int = 0
    overlap: int = 0  # found by both retrievers
    reranked: bool = False
    notes: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(
    dense: list[dict],
    sparse: list[dict],
    k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[RetrievedChunk]:
    """Merge two ranked lists into one. Input lists must be rank-ordered."""
    merged: dict[str, RetrievedChunk] = {}

    def ensure(hit: dict) -> RetrievedChunk:
        cid = hit["chunk_id"]
        if cid not in merged:
            merged[cid] = RetrievedChunk(
                chunk_id=cid, text=hit["text"], meta=hit["meta"]
            )
        return merged[cid]

    for rank, hit in enumerate(dense, start=1):
        item = ensure(hit)
        item.dense_rank = rank
        item.dense_score = hit["score"]
        item.rrf_score += dense_weight / (k + rank)

    for rank, hit in enumerate(sparse, start=1):
        item = ensure(hit)
        item.sparse_rank = rank
        item.sparse_score = hit["score"]
        item.rrf_score += sparse_weight / (k + rank)

    return sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)


class HybridRetriever:
    """Dense + sparse + fusion + optional reranking, over one strategy's index."""

    def __init__(
        self,
        settings: Settings,
        embedder: LocalEmbedder,
        strategy: str = "fixed",
        reranker=None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.strategy = strategy
        self.reranker = reranker
        self.vector_store = VectorStore(settings.chroma_dir, strategy)
        self.bm25_store = Bm25Store(settings.bm25_dir, strategy)
        self.bm25_store.load()

    def retrieve(
        self,
        query: str,
        top_n: Optional[int] = None,
        use_dense: bool = True,
        use_sparse: bool = True,
        rerank: Optional[bool] = None,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        """Return the final ranked chunks plus a trace of how they got there.

        use_dense/use_sparse exist so the dashboard's hybrid-vs-dense toggle
        and the evaluation's ablation runs share one code path -- an ablation
        that goes through different code is not measuring the same system.
        """
        top_n = top_n or self.settings.rerank_top_n
        rerank = self.settings.rerank_enabled if rerank is None else rerank
        trace = RetrievalTrace(query=query, strategy=self.strategy)

        dense_hits: list[dict] = []
        sparse_hits: list[dict] = []

        if use_dense:
            dense_hits = self.vector_store.query(
                self.embedder.embed_query(query), self.settings.dense_top_k
            )
            trace.dense_hits = len(dense_hits)
        if use_sparse:
            sparse_hits = self.bm25_store.query(query, self.settings.sparse_top_k)
            trace.sparse_hits = len(sparse_hits)

        fused = reciprocal_rank_fusion(
            dense_hits,
            sparse_hits,
            k=self.settings.rrf_k,
            dense_weight=self.settings.dense_weight,
            sparse_weight=self.settings.sparse_weight,
        )
        trace.fused_candidates = len(fused)
        trace.overlap = sum(1 for c in fused if c.found_by == "both")

        if not fused:
            trace.notes.append("no candidates from either retriever")
            return [], trace

        candidates = fused[: self.settings.rerank_candidates]

        if rerank and self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates)
            trace.reranked = True
        elif rerank and self.reranker is None:
            trace.notes.append("reranking requested but no reranker available")

        return candidates[:top_n], trace
