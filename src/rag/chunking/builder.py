"""Turns spans into labelled Chunks.

This is the single place where a chunk acquires its provenance. Every strategy
routes through here, so a citation produced by the fixed chunker and one
produced by the semantic chunker are constructed identically.
"""

from __future__ import annotations

from ..models import (
    AnchorResolver,
    Chunk,
    ChunkMetadata,
    LoadedDocument,
    make_chunk_id,
    utc_now,
)
from .base import ChunkingStrategy, Tokenizer


def build_chunks(
    doc: LoadedDocument,
    strategy: ChunkingStrategy,
    tokenizer: Tokenizer,
    min_chunk_chars: int = 80,
) -> list[Chunk]:
    resolver = AnchorResolver(doc)
    stamp = utc_now()
    chunks: list[Chunk] = []

    for ordinal, span in enumerate(strategy.spans(doc)):
        text = doc.text[span.start : span.end].strip()
        if len(text) < min_chunk_chars:
            continue

        heading, heading_path = resolver.heading_at(span.start)
        meta = ChunkMetadata(
            source=doc.source,
            source_name=doc.source_name,
            media_type=doc.media_type,
            strategy=strategy.name,
            ordinal=ordinal,
            char_start=span.start,
            char_end=span.end,
            n_tokens=tokenizer.count(text),
            heading=heading,
            heading_path=heading_path,
            page=resolver.page_at(span.start),
            ingested_at=stamp,
        )
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(doc.doc_id, strategy.name, ordinal, text),
                doc_id=doc.doc_id,
                text=text,
                meta=meta,
            )
        )

    return chunks
