"""Structure-aware chunking: cut on the document's own headings.

The premise is that a document's author already decided where the topic
boundaries are, and those decisions are recorded in its headings. Respecting
them costs nothing and tends to produce chunks that read as complete thoughts.

Two behaviours worth understanding before reading Phase 4's results:

* Sections longer than the token budget are split recursively (paragraph, then
  line, then sentence). A long section becomes several chunks that all carry
  the same heading breadcrumb.
* Consecutive short sections are merged while they fit in one chunk. This
  trades a little attribution precision -- the merged chunk cites the *first*
  section's heading -- for chunks large enough to be worth retrieving. Set
  merge_small=False to measure that tradeoff.

Documents with no headings at all (plain text) fall back to recursive splitting
over the whole document, which is the honest behaviour: there is no structure
to be aware of.
"""

from __future__ import annotations

from ..models import LoadedDocument
from .base import Span, Tokenizer
from .splitting import split_to_budget


class HeaderChunker:
    name = "header"

    def __init__(
        self,
        tokenizer: Tokenizer,
        chunk_tokens: int = 400,
        min_chars: int = 80,
        merge_small: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.chunk_tokens = chunk_tokens
        self.min_chars = min_chars
        self.merge_small = merge_small

    def spans(self, doc: LoadedDocument) -> list[Span]:
        text = doc.text
        if not text.strip():
            return []

        sections = self._sections(doc)
        if not sections:
            return split_to_budget(text, 0, len(text), self.tokenizer, self.chunk_tokens)

        if self.merge_small:
            sections = self._merge_small(text, sections)

        out: list[Span] = []
        for sec in sections:
            out.extend(
                split_to_budget(
                    text, sec.start, sec.end, self.tokenizer, self.chunk_tokens
                )
            )
        return out

    def _sections(self, doc: LoadedDocument) -> list[Span]:
        """One span per heading, running to the next heading of any level."""
        headings = sorted(doc.headings(), key=lambda a: a.offset)
        if not headings:
            return []

        bounds = [h.offset for h in headings] + [len(doc.text)]
        spans: list[Span] = []

        # Text before the first heading is its own section (front matter,
        # an intro paragraph) and would otherwise be silently dropped.
        if bounds[0] > 0 and doc.text[: bounds[0]].strip():
            spans.append(Span(0, bounds[0]))

        for start, end in zip(bounds, bounds[1:]):
            if end > start and doc.text[start:end].strip():
                spans.append(Span(start, end))
        return spans

    def _merge_small(self, text: str, sections: list[Span]) -> list[Span]:
        """Combine adjacent sections while the result still fits one chunk."""
        merged: list[Span] = []
        cur = sections[0]
        for nxt in sections[1:]:
            combined = text[cur.start : nxt.end]
            fits = self.tokenizer.count(combined) <= self.chunk_tokens
            contiguous = nxt.start == cur.end
            if fits and contiguous:
                cur = Span(cur.start, nxt.end)
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
        return merged
