"""Chunking contract.

A strategy receives the loaded document and returns character spans. It never
builds metadata, never touches a store. That restriction is what makes Phase 4's
comparison meaningful: the only thing that differs between strategies is where
the cuts fall, not how the results are labelled.

Note the input is the whole `LoadedDocument`, not just its text. Header-aware
chunking needs the anchors the loader recorded, and re-deriving headings from
the text would only work for Markdown -- HTML and PDF headings come from
structure that is gone by the time the text is flat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..models import LoadedDocument


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"empty span: {self.start}..{self.end}")

    def __len__(self) -> int:
        return self.end - self.start


class ChunkingStrategy(Protocol):
    name: str

    def spans(self, doc: LoadedDocument) -> list[Span]: ...


class Tokenizer(Protocol):
    """Minimal tokenizer surface the chunkers need."""

    def offsets(self, text: str) -> list[tuple[int, int]]: ...

    def count(self, text: str) -> int: ...
