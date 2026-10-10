"""BM25 index -- the sparse half of hybrid search.

rank_bm25 keeps everything in memory and persists nothing, so the tokenised
corpus is pickled and the index rebuilt on load. Rebuilding is fast; the
corpus is the part worth keeping on disk.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..models import Chunk

# Keep dotted/underscored identifiers intact. Config keys, error codes and
# function names are exactly what BM25 is here to catch, and a naive word split
# would shred DATABASE_URL into two useless halves.
_TOKEN = re.compile(r"[A-Za-z0-9_.\-]+")


def tokenize(text: str) -> list[str]:
    return [t.strip("._-").lower() for t in _TOKEN.findall(text) if len(t) > 1]


class Bm25Store:
    def __init__(self, persist_dir: Path, strategy: str) -> None:
        self.path = Path(persist_dir) / f"bm25_{strategy}.pkl"
        self.strategy = strategy
        self.chunk_ids: list[str] = []
        self.texts: list[str] = []
        self.metas: list[dict] = []
        self.corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    # --- persistence ---

    def load(self) -> bool:
        if not self.path.exists():
            return False
        with self.path.open("rb") as fh:
            blob = pickle.load(fh)
        self.chunk_ids = blob["chunk_ids"]
        self.texts = blob["texts"]
        self.metas = blob["metas"]
        self.corpus = blob["corpus"]
        self._reindex()
        return True

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as fh:
            pickle.dump(
                {
                    "chunk_ids": self.chunk_ids,
                    "texts": self.texts,
                    "metas": self.metas,
                    "corpus": self.corpus,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def _reindex(self) -> None:
        self._bm25 = BM25Okapi(self.corpus) if self.corpus else None

    # --- writing ---

    def add(self, chunks: list[Chunk]) -> None:
        """Upsert by chunk_id so re-ingestion does not duplicate entries."""
        existing = {cid: i for i, cid in enumerate(self.chunk_ids)}
        for c in chunks:
            row = (c.chunk_id, c.text, c.meta.as_dict(), tokenize(c.text))
            if c.chunk_id in existing:
                i = existing[c.chunk_id]
                self.chunk_ids[i], self.texts[i], self.metas[i], self.corpus[i] = row
            else:
                self.chunk_ids.append(row[0])
                self.texts.append(row[1])
                self.metas.append(row[2])
                self.corpus.append(row[3])
        self._reindex()

    # --- reading ---

    def query(self, text: str, top_k: int) -> list[dict]:
        if self._bm25 is None or not self.corpus:
            return []
        scores = self._bm25.get_scores(tokenize(text))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[dict] = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                break
            out.append(
                {
                    "chunk_id": self.chunk_ids[i],
                    "text": self.texts[i],
                    "meta": self.metas[i],
                    "score": float(scores[i]),
                }
            )
        return out

    def count(self) -> int:
        return len(self.chunk_ids)

    def reset(self) -> None:
        self.chunk_ids, self.texts, self.metas, self.corpus = [], [], [], []
        self._bm25 = None
        if self.path.exists():
            self.path.unlink()
