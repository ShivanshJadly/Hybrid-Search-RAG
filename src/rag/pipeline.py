"""Ingestion pipeline: files in, two synchronised indexes out.

Order matters here and is not arbitrary:

    load -> chunk -> exact-dedup -> embed -> near-dedup -> index

Exact deduplication runs before embedding because it is free and removes work.
Near-duplicate detection has to run *after* embedding, since it compares
vectors -- so a duplicate is embedded once before being discarded. That waste
is acceptable locally; on a paid embedding API it would be worth reordering.

The two indexes drifting apart is the failure mode this module exists to
prevent. Chroma and BM25 are written from the same post-dedup list in one pass,
and the run ends by asserting both hold the same count. Silent drift degrades
Phase 2 fusion in a way that is very hard to diagnose, because nothing raises.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .chunking.builder import build_chunks
from .chunking.registry import get_strategy
from .config import Settings
from .dedup import drop_near_duplicates
from .embeddings import LocalEmbedder
from .loaders.registry import discover, load_file
from .models import Chunk, content_fingerprint
from .stores.bm25 import Bm25Store
from .stores.documents import DocumentStore
from .stores.vector import VectorStore

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    strategy: str
    files_found: int = 0
    files_loaded: int = 0
    files_failed: list[tuple[str, str]] = field(default_factory=list)
    documents_stored: int = 0
    chunks_built: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    chunks_indexed: int = 0
    vector_count: int = 0
    bm25_count: int = 0
    empty_documents: list[str] = field(default_factory=list)
    duplicate_examples: list[tuple[str, str, float]] = field(default_factory=list)
    max_chunk_tokens: int = 0

    @property
    def in_sync(self) -> bool:
        return self.vector_count == self.bm25_count


class IndexSyncError(RuntimeError):
    pass


def ingest(
    root: Path,
    settings: Settings,
    embedder: LocalEmbedder,
    strategy_name: str = "fixed",
    rebuild: bool = False,
) -> IngestReport:
    settings.ensure_dirs()
    embedder.validate()  # refuses to run if chunks would be silently truncated

    report = IngestReport(strategy=strategy_name)
    strategy = get_strategy(
        strategy_name, embedder, settings, embed_fn=embedder.embed_passages
    )

    vector_store = VectorStore(settings.chroma_dir, strategy_name)
    bm25_store = Bm25Store(settings.bm25_dir, strategy_name)
    bm25_store.load()
    # Shared across strategies: the loaded form of a document does not depend
    # on how it is later chunked, so this is written once and re-read by the
    # other two strategy runs.
    doc_store = DocumentStore(settings.documents_dir)

    if rebuild:
        log.info("Rebuilding indexes for strategy %s", strategy_name)
        vector_store.reset()
        bm25_store.reset()

    paths = discover(root)
    report.files_found = len(paths)

    # --- load + chunk ---
    all_chunks: list[Chunk] = []
    for path in paths:
        try:
            doc = load_file(path, root)
        except Exception as exc:  # a single bad PDF must not kill the run
            log.warning("Failed to load %s: %s", path.name, exc)
            report.files_failed.append((path.name, str(exc)))
            continue

        report.files_loaded += 1

        # Persisted before the empty check, so a document that extracted to
        # nothing still leaves its raw form on disk to be diagnosed with.
        doc_store.save(doc)
        report.documents_stored += 1

        if not doc.text.strip():
            # Common with scanned PDFs that contain images rather than text.
            report.empty_documents.append(doc.source)
            continue

        all_chunks.extend(
            build_chunks(doc, strategy, embedder, settings.min_chunk_chars)
        )

    report.chunks_built = len(all_chunks)

    # --- exact duplicates (free, pre-embedding) ---
    seen: set[str] = set()
    unique: list[Chunk] = []
    for c in all_chunks:
        fp = content_fingerprint(c.text)
        if fp in seen:
            continue
        seen.add(fp)
        unique.append(c)
    report.exact_duplicates = len(all_chunks) - len(unique)

    if not unique:
        report.vector_count = vector_store.count()
        report.bm25_count = bm25_store.count()
        return report

    report.max_chunk_tokens = max(c.meta.n_tokens for c in unique)

    # --- embed, then near-duplicates ---
    vectors = embedder.embed_passages([c.text for c in unique], show_progress=True)
    deduped = drop_near_duplicates(
        unique, vectors, threshold=settings.near_duplicate_threshold
    )
    report.near_duplicates = deduped.removed
    report.duplicate_examples = deduped.examples

    # --- write both indexes from the same list ---
    vector_store.add(deduped.chunks, deduped.vectors)
    bm25_store.add(deduped.chunks)
    bm25_store.save()

    report.chunks_indexed = len(deduped.chunks)
    report.vector_count = vector_store.count()
    report.bm25_count = bm25_store.count()

    if not report.in_sync:
        raise IndexSyncError(
            f"Index drift for strategy '{strategy_name}': "
            f"Chroma has {report.vector_count} chunks, BM25 has {report.bm25_count}. "
            f"Re-run with --rebuild."
        )

    return report


def search(
    query: str,
    settings: Settings,
    embedder: LocalEmbedder,
    strategy_name: str = "fixed",
    top_k: int = 5,
) -> dict[str, list[dict]]:
    """Phase 1 acceptance check: query each index separately, no fusion yet.

    Fusion (RRF) and reranking are Phase 2. Seeing the two lists side by side is
    the point -- it shows what each half of hybrid search contributes before
    any merging hides it.
    """
    vector_store = VectorStore(settings.chroma_dir, strategy_name)
    bm25_store = Bm25Store(settings.bm25_dir, strategy_name)
    bm25_store.load()

    return {
        "dense": vector_store.query(embedder.embed_query(query), top_k),
        "sparse": bm25_store.query(query, top_k),
    }
