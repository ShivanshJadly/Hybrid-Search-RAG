"""ChromaDB wrapper -- the dense half of hybrid search.

One collection per chunking strategy. Phase 4 evaluates strategies
independently, and separate collections keep that a loop over collection names
instead of a metadata filter on every single query.
"""

from __future__ import annotations

import json

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from ..config import Settings
from ..models import Chunk


class VectorStore:
    def __init__(self, settings: Settings, strategy: str) -> None:
        self.strategy = strategy
        fields = {
            "embedding_model", "chunk_tokens", "chunk_overlap_tokens",
            "min_chunk_chars", "near_duplicate_threshold",
        }
        if strategy == "header":
            fields.add("header_merge_small")
        elif strategy == "semantic":
            fields.update({"semantic_percentile", "semantic_buffer_size"})
        self.build = json.dumps(settings.model_dump(include=fields), sort_keys=True)
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self._open()

    def _open(self):
        return self.client.get_or_create_collection(
            name=f"chunks_{self.strategy}",
            # Vectors are L2-normalised, so cosine is the right space.
            metadata={"hnsw:space": "cosine", "strategy": self.strategy, "build": self.build},
        )

    def check_build(self) -> None:
        if (self.collection.metadata or {}).get("build") != self.build:
            raise RuntimeError(
                f"The '{self.strategy}' index has different or unknown build settings. "
                f"Run 'python -m rag ingest --strategy {self.strategy} --rebuild' to rebuild it."
            )

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.check_build()
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[v.tolist() for v in vectors],
            metadatas=[c.meta.as_dict() for c in chunks],
        )

    def query(self, vector: np.ndarray, top_k: int) -> list[dict]:
        self.check_build()
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
