"""Strategy lookup.

All three strategies from the plan are live. `semantic` needs an embedding
function; the others do not, so the factory takes an optional embed_fn and only
the semantic branch requires it.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..config import Settings
from .base import ChunkingStrategy, Tokenizer
from .fixed import FixedSizeChunker
from .header import HeaderChunker
from .semantic import SemanticChunker

STRATEGIES: tuple[str, ...] = ("fixed", "header", "semantic")


class UnknownStrategy(Exception):
    pass


def get_strategy(
    name: str,
    tokenizer: Tokenizer,
    settings: Settings,
    embed_fn: Optional[Callable[[list[str]], np.ndarray]] = None,
) -> ChunkingStrategy:
    if name == "fixed":
        return FixedSizeChunker(
            tokenizer=tokenizer,
            chunk_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            min_chars=settings.min_chunk_chars,
        )

    if name == "header":
        return HeaderChunker(
            tokenizer=tokenizer,
            chunk_tokens=settings.chunk_tokens,
            min_chars=settings.min_chunk_chars,
            merge_small=settings.header_merge_small,
        )

    if name == "semantic":
        if embed_fn is None:
            raise UnknownStrategy(
                "The 'semantic' strategy needs an embedding function; "
                "pass embed_fn=embedder.embed_passages."
            )
        return SemanticChunker(
            tokenizer=tokenizer,
            embed_fn=embed_fn,
            chunk_tokens=settings.chunk_tokens,
            percentile=settings.semantic_percentile,
            buffer_size=settings.semantic_buffer_size,
        )

    raise UnknownStrategy(
        f"Unknown strategy '{name}'. Available: {', '.join(STRATEGIES)}"
    )
