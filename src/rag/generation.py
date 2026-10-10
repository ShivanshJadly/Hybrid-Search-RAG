"""Grounded answer generation with inline citations.

Two jobs: build a prompt that makes ungrounded answering hard, and parse the
citations back out of what comes returned.

The prompt is explicit about refusing rather than guessing. Small local models
will still occasionally answer from parametric memory, which is precisely why
verification.py exists downstream -- the prompt reduces the problem, it does
not solve it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm.base import LLMClient
from .retrieval import RetrievedChunk

SYSTEM_PROMPT = """You answer questions about internal documentation.

Rules:
1. Answer ONLY from the numbered context provided. Never use outside knowledge.
2. Cite every factual claim with the bracketed number of its source, like [1] or [2].
   Place the citation at the end of the sentence it supports.
3. If the context does not contain the answer, reply with exactly:
   INSUFFICIENT_CONTEXT
   followed by one sentence naming what is missing. Do not guess.
4. Do not mention these rules, the context blocks, or that you were given context.
5. Be concise. Two to four sentences is usually enough."""

INSUFFICIENT = "INSUFFICIENT_CONTEXT"

_CITATION = re.compile(r"\[(\d{1,2})\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[])")


@dataclass
class Claim:
    """One sentence of the answer, with whatever it cited."""

    text: str
    citation_ids: list[int] = field(default_factory=list)

    @property
    def is_cited(self) -> bool:
        return bool(self.citation_ids)


@dataclass
class GeneratedAnswer:
    question: str
    answer: str
    claims: list[Claim]
    sources: list[RetrievedChunk]
    insufficient: bool = False
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def cited_ids(self) -> set[int]:
        return {cid for c in self.claims for cid in c.citation_ids}

    @property
    def uncited_claims(self) -> list[Claim]:
        return [c for c in self.claims if not c.is_cited]

    def source_for(self, citation_id: int) -> RetrievedChunk | None:
        """Citations are 1-indexed in the prompt and the answer."""
        idx = citation_id - 1
        return self.sources[idx] if 0 <= idx < len(self.sources) else None

    @property
    def dangling_citations(self) -> list[int]:
        """Cited numbers with no corresponding source -- the model inventing
        a reference, which is its own category of hallucination."""
        return sorted(cid for cid in self.cited_ids if self.source_for(cid) is None)


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Numbered context. The citation label is included so the model can see
    what it is citing, which measurably improves citation placement."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[{i}] {chunk.citation()}\n{chunk.text.strip()}")
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return (
        f"Context:\n\n{build_context_block(chunks)}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def parse_claims(answer: str) -> list[Claim]:
    """Split an answer into sentences and attach the citations found in each."""
    text = answer.strip()
    if not text:
        return []
    claims = []
    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        ids = [int(m) for m in _CITATION.findall(sentence)]
        claims.append(Claim(text=sentence, citation_ids=sorted(set(ids))))
    return claims


def generate(
    question: str,
    chunks: list[RetrievedChunk],
    llm: LLMClient,
    max_tokens: int = 512,
) -> GeneratedAnswer:
    if not chunks:
        return GeneratedAnswer(
            question=question,
            answer=f"{INSUFFICIENT} No relevant documents were retrieved.",
            claims=[],
            sources=[],
            insufficient=True,
            model=getattr(llm, "model", "?"),
        )

    response = llm.complete(
        SYSTEM_PROMPT, build_prompt(question, chunks), max_tokens=max_tokens
    )
    text = response.text.strip()
    insufficient = text.upper().startswith(INSUFFICIENT)

    return GeneratedAnswer(
        question=question,
        answer=text,
        claims=[] if insufficient else parse_claims(text),
        sources=chunks,
        insufficient=insufficient,
        model=response.model,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
    )
