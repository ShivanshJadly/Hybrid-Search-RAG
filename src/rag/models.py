"""Core data structures shared across the pipeline.

The design rule here: loaders produce *text plus anchors*, chunkers decide
*where to cut*, and this module owns *how a cut is labelled*. Keeping labelling
in one place is what makes the three chunking strategies interchangeable -- a
strategy never constructs metadata itself, so strategies cannot drift apart in
how they cite sources.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

AnchorKind = Literal["heading", "page"]


@dataclass(frozen=True)
class Anchor:
    """A positional marker inside a document's cleaned text.

    Loaders emit these so that any character offset can later be resolved back
    to the heading it sits under and the page it came from. This is the whole
    basis of Phase 3 citations: if it is not captured at load time, it cannot
    be recovered without re-ingesting.
    """

    offset: int
    kind: AnchorKind
    value: str
    level: int = 0  # heading depth; 0 for pages


@dataclass
class LoadedDocument:
    """One source file, cleaned into plain text with positional anchors.

    Both forms are kept. `text` is what everything downstream uses; `raw_text`
    is what the extractor produced before normalisation, and exists so that
    extraction quality can be audited without re-parsing. For HTML that is the
    original markup, which is the only way to see what the parser discarded;
    for PDF it is the un-normalised page text, which is where extraction
    artefacts show up.
    """

    doc_id: str
    source: str        # path relative to the ingest root
    source_name: str   # bare filename, for display
    media_type: str    # md | pdf | html | txt
    text: str
    raw_text: str = ""
    anchors: list[Anchor] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def headings(self) -> list[Anchor]:
        return [a for a in self.anchors if a.kind == "heading"]

    def pages(self) -> list[Anchor]:
        return [a for a in self.anchors if a.kind == "page"]


@dataclass
class ChunkMetadata:
    """Everything needed to cite a chunk. Flat by design -- Chroma only
    accepts scalar metadata values."""

    source: str
    source_name: str
    media_type: str
    strategy: str
    ordinal: int
    char_start: int
    char_end: int
    n_tokens: int
    heading: str = ""       # nearest preceding heading
    heading_path: str = ""  # breadcrumb, e.g. "Deploy Guide > Rollbacks"
    page: int = 0           # 1-indexed; 0 means not applicable
    ingested_at: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    meta: ChunkMetadata

    def citation(self) -> str:
        """Human-readable provenance string, used by the search CLI today and
        by the answer formatter in Phase 3."""
        bits = [self.meta.source_name]
        if self.meta.heading_path:
            bits.append(self.meta.heading_path)
        if self.meta.page:
            bits.append(f"p.{self.meta.page}")
        return " > ".join(bits)


class AnchorResolver:
    """Answers 'which heading and page is character offset N under?'

    Built once per document; lookups are binary searches over sorted offsets.
    """

    def __init__(self, doc: LoadedDocument) -> None:
        headings = sorted(doc.headings(), key=lambda a: a.offset)
        pages = sorted(doc.pages(), key=lambda a: a.offset)
        self._h_offsets = [a.offset for a in headings]
        self._headings = headings
        self._p_offsets = [a.offset for a in pages]
        self._pages = pages

    def heading_at(self, offset: int) -> tuple[str, str]:
        """Return (heading, breadcrumb) for the given offset."""
        i = bisect_right(self._h_offsets, offset) - 1
        if i < 0:
            return "", ""
        current = self._headings[i]
        # Walk backwards collecting strictly shallower headings to form a path.
        trail = [current]
        level = current.level
        for prev in reversed(self._headings[:i]):
            if prev.level < level:
                trail.append(prev)
                level = prev.level
            if level <= 1:
                break
        trail.reverse()
        return current.value, " > ".join(a.value for a in trail)

    def page_at(self, offset: int) -> int:
        i = bisect_right(self._p_offsets, offset) - 1
        if i < 0:
            return 0
        try:
            return int(self._pages[i].value)
        except ValueError:
            return 0


def make_doc_id(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def make_chunk_id(doc_id: str, strategy: str, ordinal: int, text: str) -> str:
    """Deterministic so that re-ingesting an unchanged corpus is idempotent
    rather than duplicating every chunk."""
    payload = f"{doc_id}|{strategy}|{ordinal}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def content_fingerprint(text: str) -> str:
    """Whitespace/case-insensitive hash, used for exact-duplicate detection."""
    normalised = " ".join(text.lower().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
