"""Near-duplicate removal.

Internal documentation repeats itself: licence headers, shared boilerplate,
the same warning pasted into four runbooks. Every duplicate that survives into
the index competes for one of the handful of context slots the generator gets,
so a retrieved set of five can easily be the same paragraph five times.

Exact duplicates are caught earlier by content hash. This catches the harder
case -- text that differs by a word or two but says the same thing.

Method: greedy, order-preserving. Walk the chunks in order and keep one only if
it is not too similar to anything already kept. First occurrence wins, so the
result is deterministic for a given input order.

Cost: O(n*k) dot products for n chunks and k kept, done as blocked matrix
multiplications. Fine into the tens of thousands of chunks. Beyond that this
should become an approximate-neighbour lookup rather than a scan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Chunk


@dataclass
class DedupResult:
    chunks: list[Chunk]
    vectors: np.ndarray
    removed: int
    # (kept citation, dropped citation, similarity) -- useful for reporting
    # and for sanity-checking that the threshold is not too aggressive.
    examples: list[tuple[str, str, float]]


def drop_near_duplicates(
    chunks: list[Chunk],
    vectors: np.ndarray,
    threshold: float = 0.95,
    max_examples: int = 5,
) -> DedupResult:
    """Return the chunks and vectors that survive deduplication.

    Vectors must be row-aligned with chunks. They arrive L2-normalised from the
    embedder, so a dot product is the cosine similarity; we renormalise anyway
    so this stays correct if that ever changes upstream.
    """
    if len(chunks) == 0:
        return DedupResult([], vectors, 0, [])
    if len(chunks) != len(vectors):
        raise ValueError(
            f"chunks ({len(chunks)}) and vectors ({len(vectors)}) are not aligned"
        )
    if len(chunks) == 1:
        return DedupResult(list(chunks), vectors, 0, [])

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms

    kept_idx: list[int] = []
    examples: list[tuple[str, str, float]] = []

    for i in range(len(chunks)):
        if not kept_idx:
            kept_idx.append(i)
            continue
        sims = unit[kept_idx] @ unit[i]
        best = int(np.argmax(sims))
        if float(sims[best]) >= threshold:
            if len(examples) < max_examples:
                examples.append(
                    (
                        chunks[kept_idx[best]].citation(),
                        chunks[i].citation(),
                        float(sims[best]),
                    )
                )
            continue
        kept_idx.append(i)

    return DedupResult(
        chunks=[chunks[i] for i in kept_idx],
        vectors=vectors[kept_idx],
        removed=len(chunks) - len(kept_idx),
        examples=examples,
    )
