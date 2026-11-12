"""ChromaDB wrapper -- the dense half of hybrid search.

One collection per chunking strategy. Phase 4 evaluates strategies
independently, and separate collections keep that a loop over collection names
instead of a metadata filter on every single query.
"""

from __future__ import annotations

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from ..models import Chunk, make_doc_id


class VectorStore:
    def __init__(self, persist_dir, strategy: str) -> None:
        self.strategy = strategy
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self._open()

    def _open(self):
        return self.client.get_or_create_collection(
            name=f"chunks_{self.strategy}",
            # Vectors are L2-normalised, so cosine is the right space.
            metadata={"hnsw:space": "cosine", "strategy": self.strategy},
        )

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[v.tolist() for v in vectors],
            metadatas=[{**c.meta.as_dict(), "doc_id": c.doc_id} for c in chunks],
        )

    def delete(self, doc_id: str, source: str | None = None) -> int:
        count_before = self.count()
        if source is not None:
            self.collection.delete(where={"source": source})
        else:
            records = self.collection.get(include=["metadatas"])
            chunk_ids = [
                chunk_id
                for chunk_id, metadata in zip(records["ids"], records["metadatas"])
                if metadata.get("doc_id") == doc_id
                or (
                    metadata.get("source")
                    and make_doc_id(metadata["source"]) == doc_id
                )
            ]
            if chunk_ids:
                self.collection.delete(ids=chunk_ids)
        return count_before - self.count()

    def query(self, vector: np.ndarray, top_k: int) -> list[dict]:
        res = self.collection.query(
            query_embeddings=[vector.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if not res["ids"] or not res["ids"][0]:
            return []
        out: list[dict] = []
        for i, cid in enumerate(res["ids"][0]):
            distance = res["distances"][0][i]
            out.append(
                {
                    "chunk_id": cid,
                    "text": res["documents"][0][i],
                    "meta": res["metadatas"][0][i],
                    # Chroma returns cosine *distance*; similarity reads better.
                    "score": 1.0 - float(distance),
                }
            )
        return out

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection.name)
        except Exception:
            pass
        self.collection = self._open()
