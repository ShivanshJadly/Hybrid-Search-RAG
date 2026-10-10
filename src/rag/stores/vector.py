"""ChromaDB wrapper -- the dense half of hybrid search.

One collection per chunking strategy. Phase 4 evaluates strategies
independently, and separate collections keep that a loop over collection names
instead of a metadata filter on every single query.
"""

from __future__ import annotations

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from ..models import Chunk


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
            metadatas=[c.meta.as_dict() for c in chunks],
        )

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
