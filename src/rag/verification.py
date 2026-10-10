"""Citation verification -- the step most RAG systems skip.

Generating `[1]` next to a sentence proves nothing: the model may cite a chunk
that does not actually support the claim, or cite a number that does not exist.
This module re-reads each claim against the chunk it points at and asks a judge
whether the support is real.

The judge is a separate backend from the generator on purpose (Notion plan
A.4): asking a model to grade its own output is the weakest possible check, and
grading is exactly where a small local model underperforms.

Verdicts are deliberately coarse -- SUPPORTED / PARTIAL / UNSUPPORTED. Finer
scales produce judge disagreement without producing more signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .generation import Claim, GeneratedAnswer
from .llm.base import LLMClient

JUDGE_SYSTEM = """You check whether a claim is supported by a source passage.

Reply with exactly one word on the first line:
SUPPORTED   - the passage states or directly implies the claim
PARTIAL     - the passage is related but does not fully establish the claim
UNSUPPORTED - the passage does not support the claim

Then one short sentence of justification on the next line.
Judge only against the passage given. Ignore whether the claim is true in general."""


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"
    DANGLING = "DANGLING"  # cited a source number that does not exist
    UNCITED = "UNCITED"    # factual-looking sentence with no citation at all

    @property
    def is_ok(self) -> bool:
        return self is Verdict.SUPPORTED


@dataclass
class CitationCheck:
    claim: str
    citation_id: int
    verdict: Verdict
    reason: str = ""
    source_citation: str = ""


@dataclass
class VerificationReport:
    checks: list[CitationCheck]
    judge_model: str = ""

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def supported(self) -> int:
        return sum(1 for c in self.checks if c.verdict is Verdict.SUPPORTED)

    @property
    def problems(self) -> list[CitationCheck]:
        return [c for c in self.checks if not c.verdict.is_ok]

    @property
    def measured(self) -> bool:
        """Whether any citation was actually checked.

        A refusal cites nothing, so there is nothing to verify. Callers that
        report accuracy to a human need to tell "everything held up" apart
        from "nothing was examined"; `accuracy` alone cannot.
        """
        return self.total > 0

    @property
    def accuracy(self) -> float:
        """Share of citations that hold up. 1.0 when there is nothing to check
        -- an answer that cited nothing is punished by coverage, not here.
        Check `measured` before presenting this as a result."""
        return self.supported / self.total if self.total else 1.0


_VERDICT_RE = re.compile(r"\b(SUPPORTED|PARTIAL|UNSUPPORTED)\b", re.I)


def _parse_verdict(text: str) -> tuple[Verdict, str]:
    match = _VERDICT_RE.search(text or "")
    if not match:
        # An unparseable judge reply must not silently count as a pass.
        return Verdict.PARTIAL, "judge reply could not be parsed"
    verdict = Verdict(match.group(1).upper())
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    reason = lines[1] if len(lines) > 1 else ""
    return verdict, reason


def _looks_factual(claim: Claim) -> bool:
    """Filter out hedges and pleasantries before demanding a citation.

    Without this, 'I hope that helps.' counts as an uncited claim and drags
    coverage down for no reason.
    """
    text = claim.text.strip().lower()
    if len(text.split()) < 4:
        return False
    openers = ("i hope", "let me know", "in summary", "to summarise", "note that")
    return not text.startswith(openers)


def verify(answer: GeneratedAnswer, judge: LLMClient) -> VerificationReport:
    checks: list[CitationCheck] = []

    if answer.insufficient:
        return VerificationReport(checks=[], judge_model=getattr(judge, "model", "?"))

    for claim in answer.claims:
        if not claim.is_cited:
            if _looks_factual(claim):
                checks.append(
                    CitationCheck(
                        claim=claim.text,
                        citation_id=0,
                        verdict=Verdict.UNCITED,
                        reason="factual sentence carries no citation",
                    )
                )
            continue

        for cid in claim.citation_ids:
            source = answer.source_for(cid)
            if source is None:
                checks.append(
                    CitationCheck(
                        claim=claim.text,
                        citation_id=cid,
                        verdict=Verdict.DANGLING,
                        reason=f"cited [{cid}] but only {len(answer.sources)} sources exist",
                    )
                )
                continue

            prompt = (
                f"Passage:\n{source.text.strip()}\n\n"
                f"Claim:\n{claim.text.strip()}\n\nVerdict:"
            )
            reply = judge.complete(JUDGE_SYSTEM, prompt, max_tokens=128)
            verdict, reason = _parse_verdict(reply.text)
            checks.append(
                CitationCheck(
                    claim=claim.text,
                    citation_id=cid,
                    verdict=verdict,
                    reason=reason,
                    source_citation=source.citation(),
                )
            )

    return VerificationReport(checks=checks, judge_model=getattr(judge, "model", "?"))
