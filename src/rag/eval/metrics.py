"""Evaluation metrics.

Four metrics, matching the plan, deliberately measuring different layers:

* **retrieval relevance** -- did the right document reach the context window?
  Computed from document names, not model output, so it is completely
  independent of generation quality. When this is low, nothing downstream can
  be fixed by prompting.
* **correctness** -- does the answer match the golden answer? Needs a judge.
* **faithfulness** -- is the answer grounded in what was retrieved, regardless
  of whether it is correct? An answer can be correct from parametric memory
  and unfaithful, which is a hallucination that happened to be right.
* **citation accuracy** -- do the inline citations hold up? Reuses the same
  verification pass production runs.

Plus **refusal accuracy** for the no-answer category, which is a pass/fail on
whether the system declined when it should have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from ..answering import Answer
from ..llm.base import LLMClient
from ..verification import Verdict
from .dataset import GoldenQuestion

CORRECTNESS_SYSTEM = """You grade an answer against a reference answer.

Reply with exactly one word on the first line:
CORRECT   - conveys the same key facts as the reference
PARTIAL   - some key facts right, others missing or wrong
INCORRECT - contradicts the reference or misses its substance

Then one short sentence of justification.
Wording may differ freely. Grade the facts, not the phrasing."""

FAITHFULNESS_SYSTEM = """You check whether an answer is grounded in the given context.

Reply with exactly one word on the first line:
GROUNDED   - every factual statement traces to the context
PARTIAL    - mostly grounded, but contains at least one unsupported statement
UNGROUNDED - substantially not supported by the context

Then one short sentence of justification.
Do not consider whether the answer is true in general, only whether the
context supports it."""


@dataclass
class QuestionResult:
    id: str
    category: str
    question: str
    answerable: bool

    retrieval_hit: bool = False
    retrieval_precision: float = 0.0
    correctness: str = "SKIPPED"
    faithfulness: str = "SKIPPED"
    citation_accuracy: float = 1.0
    declined: bool = False
    refusal_correct: bool | None = None
    confidence: float = 0.0
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.correctness == "CORRECT"

    @property
    def faithful(self) -> bool:
        return self.faithfulness == "GROUNDED"


@dataclass
class EvalSummary:
    strategy: str
    results: list[QuestionResult]
    generator: str = ""
    judge: str = ""

    def _answerable(self) -> list[QuestionResult]:
        return [r for r in self.results if r.answerable]

    @property
    def retrieval_recall(self) -> float:
        rows = self._answerable()
        return mean(1.0 if r.retrieval_hit else 0.0 for r in rows) if rows else 0.0

    @property
    def retrieval_precision(self) -> float:
        rows = self._answerable()
        return mean(r.retrieval_precision for r in rows) if rows else 0.0

    @property
    def correctness_rate(self) -> float:
        rows = [r for r in self._answerable() if r.correctness != "SKIPPED"]
        return mean(1.0 if r.correct else 0.0 for r in rows) if rows else 0.0

    @property
    def faithfulness_rate(self) -> float:
        rows = [r for r in self.results if r.faithfulness != "SKIPPED"]
        return mean(1.0 if r.faithful else 0.0 for r in rows) if rows else 0.0

    @property
    def citation_accuracy(self) -> float:
        rows = self._answerable()
        return mean(r.citation_accuracy for r in rows) if rows else 0.0

    @property
    def refusal_accuracy(self) -> float:
        """Of the questions the corpus cannot answer, how many were declined."""
        rows = [r for r in self.results if not r.answerable]
        return mean(1.0 if r.refusal_correct else 0.0 for r in rows) if rows else 0.0

    @property
    def false_refusal_rate(self) -> float:
        """Answerable questions the system wrongly declined. The metric that
        stops 'always decline' from looking like a good strategy."""
        rows = self._answerable()
        return mean(1.0 if r.declined else 0.0 for r in rows) if rows else 0.0

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "generator": self.generator,
            "judge": self.judge,
            "questions": len(self.results),
            "retrieval_recall": round(self.retrieval_recall, 4),
            "retrieval_precision": round(self.retrieval_precision, 4),
            "correctness": round(self.correctness_rate, 4),
            "faithfulness": round(self.faithfulness_rate, 4),
            "citation_accuracy": round(self.citation_accuracy, 4),
            "refusal_accuracy": round(self.refusal_accuracy, 4),
            "false_refusal_rate": round(self.false_refusal_rate, 4),
        }


def score_retrieval(answer: Answer, golden: GoldenQuestion) -> tuple[bool, float]:
    """Recall-style hit plus precision over the retrieved set.

    Matching on source_name rather than chunk id: the golden set should not
    have to be rewritten every time chunking changes, which would make
    cross-strategy comparison impossible.
    """
    if not golden.expected_sources:
        return False, 0.0
    retrieved = [s.meta.get("source_name", "") for s in answer.sources]
    if not retrieved:
        return False, 0.0
    expected = set(golden.expected_sources)
    hits = sum(1 for name in retrieved if name in expected)
    return hits > 0, hits / len(retrieved)


def _first_word(text: str, allowed: tuple[str, ...], fallback: str) -> str:
    upper = (text or "").upper()
    for token in allowed:
        if token in upper:
            return token
    return fallback


def judge_correctness(answer: Answer, golden: GoldenQuestion, judge: LLMClient) -> str:
    if not golden.expected_answer:
        return "SKIPPED"
    reply = judge.complete(
        CORRECTNESS_SYSTEM,
        f"Question:\n{golden.question}\n\n"
        f"Reference answer:\n{golden.expected_answer}\n\n"
        f"Answer to grade:\n{answer.text}\n\nGrade:",
        max_tokens=128,
    )
    return _first_word(reply.text, ("CORRECT", "PARTIAL", "INCORRECT"), "INCORRECT")


def judge_faithfulness(answer: Answer, judge: LLMClient) -> str:
    if answer.declined or not answer.sources:
        return "SKIPPED"
    context = "\n\n".join(
        f"[{i}] {s.text}" for i, s in enumerate(answer.sources, start=1)
    )
    reply = judge.complete(
        FAITHFULNESS_SYSTEM,
        f"Context:\n{context}\n\nAnswer:\n{answer.text}\n\nGrade:",
        max_tokens=128,
    )
    return _first_word(reply.text, ("GROUNDED", "PARTIAL", "UNGROUNDED"), "UNGROUNDED")


def citation_accuracy(answer: Answer) -> float:
    checks = answer.verification.checks
    if not checks:
        return 1.0
    ok = sum(1 for c in checks if c.verdict is Verdict.SUPPORTED)
    return ok / len(checks)
