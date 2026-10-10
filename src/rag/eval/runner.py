"""Evaluation runner.

Runs the golden set through the *production* answer pipeline -- not a
reimplementation of it -- and scores the results. Supports sweeping across
chunking strategies, which is the plan's Phase 4.3 comparison.

Results are written to JSON so runs can be diffed. A metric with no baseline
to compare against is a number, not a finding.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..answering import AnswerPipeline
from ..config import Settings
from ..embeddings import LocalEmbedder
from ..llm.base import LLMClient
from ..rerank import build_reranker
from .dataset import GoldenQuestion, load_golden_set
from .metrics import (
    EvalSummary,
    QuestionResult,
    citation_accuracy,
    judge_correctness,
    judge_faithfulness,
    score_retrieval,
)

log = logging.getLogger(__name__)


def evaluate(
    settings: Settings,
    embedder: LocalEmbedder,
    llm: LLMClient,
    judge: LLMClient,
    strategy: str = "fixed",
    questions: Optional[Iterable[GoldenQuestion]] = None,
    limit: Optional[int] = None,
    grade: bool = True,
    on_progress: Optional[Callable[[int, int, GoldenQuestion], None]] = None,
) -> EvalSummary:
    """Score one strategy against the golden set.

    `grade=False` skips the two judge-backed metrics (correctness and
    faithfulness) and keeps the ones computed without a model. Useful for a
    fast retrieval-only sweep, and for staying inside a free-tier quota.
    """
    golden = list(questions if questions is not None else load_golden_set())
    if limit:
        golden = golden[:limit]

    pipeline = AnswerPipeline(
        settings,
        embedder,
        llm,
        judge,
        strategy=strategy,
        reranker=build_reranker(settings),
    )

    results: list[QuestionResult] = []
    for i, q in enumerate(golden, start=1):
        if on_progress:
            on_progress(i, len(golden), q)

        answer = pipeline.ask(q.question)
        hit, precision = score_retrieval(answer, q)

        result = QuestionResult(
            id=q.id,
            category=q.category,
            question=q.question,
            answerable=q.answerable,
            retrieval_hit=hit,
            retrieval_precision=precision,
            citation_accuracy=citation_accuracy(answer),
            declined=answer.declined,
            confidence=answer.confidence.composite,
            answer=answer.text,
            sources=[s.citation() for s in answer.sources],
        )

        if not q.answerable:
            # The only correct behaviour is to decline.
            result.refusal_correct = answer.declined
            if not answer.declined:
                result.notes.append("answered a question the corpus cannot support")
        elif grade:
            result.correctness = judge_correctness(answer, q, judge)
            result.faithfulness = judge_faithfulness(answer, judge)

        results.append(result)

    return EvalSummary(
        strategy=strategy,
        results=results,
        generator=getattr(llm, "model", "?"),
        judge=getattr(judge, "model", "?"),
    )


def sweep(
    settings: Settings,
    embedder: LocalEmbedder,
    llm: LLMClient,
    judge: LLMClient,
    strategies: Iterable[str],
    **kwargs,
) -> list[EvalSummary]:
    """Run the same golden set across several chunking strategies."""
    out = []
    for strategy in strategies:
        log.info("Evaluating strategy %s", strategy)
        out.append(evaluate(settings, embedder, llm, judge, strategy=strategy, **kwargs))
    return out


def write_report(summaries: list[EvalSummary], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"eval-{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": stamp,
                "summaries": [
                    {
                        **s.as_dict(),
                        "results": [asdict(r) for r in s.results],
                    }
                    for s in summaries
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
