"""The full question-answering pipeline.

    question
      -> retrieve   (dense + sparse -> RRF -> rerank)
      -> generate   (grounded prompt, inline citations)
      -> verify     (judge each claim against its cited source)
      -> score      (retrieval + coverage + citation accuracy)
      -> answer, or decline if confidence is too low

This is the single entry point used by the CLI, the API and the evaluation
runner. Having one path matters: an evaluation that exercises different code
from production is measuring something other than production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .confidence import ConfidenceBreakdown
from .confidence import score as score_confidence
from .config import Settings
from .embeddings import LocalEmbedder
from .generation import GeneratedAnswer, generate
from .llm.base import LLMClient
from .rerank import build_reranker
from .retrieval import HybridRetriever, RetrievalTrace, RetrievedChunk
from .verification import VerificationReport, verify

DECLINE_TEXT = (
    "I could not find enough supporting information in the indexed documents "
    "to answer this confidently."
)


@dataclass
class Answer:
    question: str
    text: str
    declined: bool
    confidence: ConfidenceBreakdown
    sources: list[RetrievedChunk]
    generation: GeneratedAnswer
    verification: VerificationReport
    trace: RetrievalTrace

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "declined": self.declined,
            "confidence": self.confidence.as_dict(),
            "sources": [
                {
                    "n": i,
                    "chunk_id": s.chunk_id,
                    "citation": s.citation(),
                    "source": s.meta.get("source", ""),
                    "heading_path": s.meta.get("heading_path", ""),
                    "page": s.meta.get("page", 0),
                    "found_by": s.found_by,
                    "dense_rank": s.dense_rank,
                    "sparse_rank": s.sparse_rank,
                    "rrf_score": round(s.rrf_score, 6),
                    "rerank_score": s.rerank_score,
                    "text": s.text,
                }
                for i, s in enumerate(self.sources, start=1)
            ],
            "citations": [
                {
                    "claim": c.claim,
                    "citation_id": c.citation_id,
                    "verdict": c.verdict.value,
                    "reason": c.reason,
                    "source": c.source_citation,
                }
                for c in self.verification.checks
            ],
            "retrieval": {
                "strategy": self.trace.strategy,
                "dense_hits": self.trace.dense_hits,
                "sparse_hits": self.trace.sparse_hits,
                "fused_candidates": self.trace.fused_candidates,
                "overlap": self.trace.overlap,
                "reranked": self.trace.reranked,
                "notes": self.trace.notes,
            },
            "model": self.generation.model,
            "judge": self.verification.judge_model,
        }


class AnswerPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: LocalEmbedder,
        llm: LLMClient,
        judge: LLMClient,
        strategy: str = "fixed",
        reranker=None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.judge = judge
        self.retriever = HybridRetriever(
            settings,
            embedder,
            strategy,
            reranker=reranker if reranker is not None else build_reranker(settings),
        )

    def ask(
        self,
        question: str,
        top_n: Optional[int] = None,
        use_dense: bool = True,
        use_sparse: bool = True,
        rerank: Optional[bool] = None,
        verify_citations: bool = True,
    ) -> Answer:
        chunks, trace = self.retriever.retrieve(
            question,
            top_n=top_n or self.settings.rerank_top_n,
            use_dense=use_dense,
            use_sparse=use_sparse,
            rerank=rerank,
        )

        generated = generate(question, chunks, self.llm)

        report = (
            verify(generated, self.judge)
            if verify_citations
            else VerificationReport(checks=[], judge_model="skipped")
        )

        breakdown = score_confidence(generated, chunks, report)

        # Declining is a feature, not a failure. Returning what was and was not
        # found beats returning a confident wrong answer.
        declined = (
            generated.insufficient
            or breakdown.composite < self.settings.min_answer_confidence
        )

        text = generated.answer
        if declined and not generated.insufficient:
            found = ", ".join(dict.fromkeys(s.meta.get("source_name", "?") for s in chunks))
            text = f"{DECLINE_TEXT} Closest material found in: {found or 'nothing'}."
            # The components were measured against an answer the user will
            # never see. Report the refusal's own question instead.
            breakdown = breakdown.as_refusal()

        return Answer(
            question=question,
            text=text,
            declined=declined,
            confidence=breakdown,
            sources=chunks,
            generation=generated,
            verification=report,
            trace=trace,
        )
