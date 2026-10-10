"""Composite answer confidence.

Three independent signals, because each catches a different failure:

* **retrieval** -- did we find anything relevant? Catches the question the
  corpus cannot answer.
* **coverage** -- is the answer actually cited? Catches the model asserting
  things without pointing at a source.
* **citation accuracy** -- do the citations hold up under checking? Catches
  the model citing a source that does not say what it claims.

An answer can score well on two and badly on the third, which is why they are
reported separately as well as combined. The composite exists to drive one
decision -- answer or decline -- and the breakdown exists so a human can see
why.

The weights are a starting point, not a measured result. Phase 4 is where they
get tuned against the golden set.

**Refusals are scored on their own scale.** Blending the three signals only
makes sense for an answer, because a refusal makes no claims and cites no
sources -- coverage and accuracy have nothing to measure, and pinning them at
1.0 to keep the formula whole inverts it: a refusal issued *despite* strong
retrieval then outscores a correct one, which is exactly backwards. For a
refusal the only question worth asking is whether refusing was right, and
weak retrieval is the evidence that it was.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .generation import GeneratedAnswer
from .retrieval import RetrievedChunk
from .verification import VerificationReport

RETRIEVAL_WEIGHT = 0.35
COVERAGE_WEIGHT = 0.25
CITATION_WEIGHT = 0.40


@dataclass
class ConfidenceBreakdown:
    """Scored components plus the one number that drives answer-or-decline.

    `coverage` and `citation_accuracy` are None when there was nothing of that
    kind to measure. They are reported that way rather than defaulted, so a
    reader can tell an earned score from an empty one.

    `kind` says which scale `composite` is on: "answer" is how good the answer
    is, "refusal" is how sure we are that refusing was right. The two are not
    comparable and are labelled differently for that reason.
    """

    retrieval: float
    coverage: float | None
    citation_accuracy: float | None
    composite: float
    reasons: list[str]
    kind: str = "answer"

    @property
    def label(self) -> str:
        if self.composite >= 0.75:
            return "high"
        if self.composite >= 0.5:
            return "medium"
        if self.composite >= 0.35:
            return "low"
        return "very low"

    def as_refusal(self) -> "ConfidenceBreakdown":
        """Recast onto the refusal scale.

        Used when the system suppresses an answer it did generate: the
        components were measured against an answer nobody will see, so the
        number reported alongside the refusal must ask the refusal's question
        instead. Already-refused breakdowns are returned unchanged.
        """
        if self.kind == "refusal":
            return self
        return _refusal_breakdown(self.retrieval, list(self.reasons))

    def as_dict(self) -> dict:
        return {
            "retrieval": round(self.retrieval, 4),
            "coverage": None if self.coverage is None else round(self.coverage, 4),
            "citation_accuracy": (
                None
                if self.citation_accuracy is None
                else round(self.citation_accuracy, 4)
            ),
            "composite": round(self.composite, 4),
            "kind": self.kind,
            "label": self.label,
            "reasons": self.reasons,
        }


def _squash(x: float) -> float:
    """Map a cross-encoder logit onto 0..1.

    Reranker scores are raw logits, roughly -11..+11 for this model. They order
    results correctly but mean nothing as absolute numbers, so a logistic is
    the honest way to get a bounded value out of one.
    """
    return 1.0 / (1.0 + math.exp(-x))


def retrieval_confidence(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    top = chunks[0]
    if top.rerank_score is not None:
        return _squash(top.rerank_score)
    if top.dense_score is not None:
        # Cosine similarity on normalised vectors: already 0..1, but the
        # useful band for this model sits around 0.5-0.9, so stretch it.
        return max(0.0, min(1.0, (top.dense_score - 0.35) / 0.5))
    # Fusion-only: RRF scores are tiny by construction, so this is a weak
    # signal and is treated as such.
    return min(1.0, top.rrf_score * 40)


def citation_coverage(answer: GeneratedAnswer) -> float | None:
    """Share of the answer's sentences that carry a citation.

    None when the answer declines: a refusal asserts nothing, so there is no
    coverage to measure. An answer that produced no parseable claims is a
    different thing -- generation went wrong -- and scores zero.
    """
    if answer.insufficient:
        return None
    if not answer.claims:
        return 0.0
    cited = sum(1 for c in answer.claims if c.is_cited)
    return cited / len(answer.claims)


def _blend(parts: list[tuple[float, float | None]]) -> float:
    """Weighted mean over the components that were actually measured.

    Renormalising across the surviving weights keeps the result on a 0..1
    scale when a component drops out, rather than quietly capping it at the
    sum of the weights that remain.
    """
    measured = [(w, v) for w, v in parts if v is not None]
    total = sum(w for w, _ in measured)
    if not total:
        return 0.0
    return sum(w * v for w, v in measured) / total


def _refusal_breakdown(retrieval: float, reasons: list[str]) -> "ConfidenceBreakdown":
    """Confidence that refusing was the right call.

    Weak retrieval is the evidence for a refusal, so confidence runs inverse
    to it. This is what makes a false refusal visible: decline while holding
    strong material and the number collapses, where the blended formula used
    to reward it.
    """
    composite = 1.0 - retrieval
    if retrieval >= 0.4:
        reasons.append(
            "refused despite relevant material - possible false refusal"
        )
    return ConfidenceBreakdown(
        retrieval=retrieval,
        coverage=None,
        citation_accuracy=None,
        composite=composite,
        reasons=reasons,
        kind="refusal",
    )


def score(
    answer: GeneratedAnswer,
    chunks: list[RetrievedChunk],
    report: VerificationReport,
) -> ConfidenceBreakdown:
    reasons: list[str] = []

    retrieval = retrieval_confidence(chunks)
    coverage = citation_coverage(answer)
    accuracy = report.accuracy if report.measured else None

    if retrieval < 0.4:
        reasons.append("weak retrieval: top chunk scored poorly against the question")

    # An honest refusal is not a low-confidence answer; it is a correct
    # non-answer. It gets its own scale rather than a blend of components it
    # never produced -- see the module docstring.
    if answer.insufficient:
        reasons.append("model reported insufficient context")
        return _refusal_breakdown(retrieval, reasons)

    if coverage is not None and coverage < 0.6:
        reasons.append(f"{len(answer.uncited_claims)} sentence(s) carry no citation")
    if report.problems:
        reasons.append(f"{len(report.problems)} citation(s) failed verification")
    if answer.dangling_citations:
        reasons.append(
            f"cited non-existent source(s): "
            f"{', '.join(f'[{c}]' for c in answer.dangling_citations)}"
        )

    composite = _blend(
        [
            (RETRIEVAL_WEIGHT, retrieval),
            (COVERAGE_WEIGHT, coverage),
            (CITATION_WEIGHT, accuracy),
        ]
    )

    if not reasons:
        reasons.append("all citations verified against retrieved sources")

    return ConfidenceBreakdown(
        retrieval=retrieval,
        coverage=coverage,
        citation_accuracy=accuracy,
        composite=composite,
        reasons=reasons,
    )
