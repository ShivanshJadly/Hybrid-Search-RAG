"""BM25 index -- the sparse half of hybrid search.

rank_bm25 keeps everything in memory and persists nothing, so the tokenised
corpus is pickled and the index rebuilt on load. Rebuilding is fast; the
corpus is the part worth keeping on disk.
"""

from __future__ import annotations

import os
import pickle
import re
import tempfile
import time
from pathlib import Path
from threading import RLock

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
        self._loaded_signature: tuple[int, int, int, int] | None = None
        self._lock = RLock()

    # --- persistence ---

    @staticmethod
    def _signature(stat: os.stat_result) -> tuple[int, int, int, int]:
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def _path_signature(self) -> tuple[int, int, int, int] | None:
        try:
            return self._signature(self.path.stat())
        except FileNotFoundError:
            return None

    def _clear(self) -> None:
        self.chunk_ids = []
        self.texts = []
        self.metas = []
        self.corpus = []
        self._bm25 = None
        self._loaded_signature = None

    def load(self) -> bool:
        with self._lock:
            while True:
                try:
                    with self.path.open("rb") as fh:
                        signature = self._signature(os.fstat(fh.fileno()))
                        blob = pickle.load(fh)
                except FileNotFoundError:
                    self._clear()
                    return False

                chunk_ids = blob["chunk_ids"]
                texts = blob["texts"]
                metas = blob["metas"]
                corpus = blob["corpus"]
                bm25 = BM25Okapi(corpus) if corpus else None

                # If a writer replaced the file while it was being read, load
                # the current complete snapshot before publishing any fields.
                if self._path_signature() != signature:
                    continue

                self.chunk_ids = chunk_ids
                self.texts = texts
                self.metas = metas
                self.corpus = corpus
                self._bm25 = bm25
                self._loaded_signature = signature
                return True

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=self.path.parent,
                    prefix=f".{self.path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as fh:
                    temp_path = Path(fh.name)
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
                    fh.flush()
                    os.fsync(fh.fileno())
                signature = self._signature(temp_path.stat())
                for attempt in range(4):
                    try:
                        os.replace(temp_path, self.path)
                        break
                    except PermissionError as exc:
                        if (
                            getattr(exc, "winerror", None) not in {5, 32, 33}
                            or attempt == 3
                        ):
                            raise
                        time.sleep(0.01 * (2**attempt))
                self._loaded_signature = signature
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

    def _reindex(self) -> None:
        self._bm25 = BM25Okapi(self.corpus) if self.corpus else None

    # --- writing ---

    def add(self, chunks: list[Chunk]) -> None:
        """Upsert by chunk_id so re-ingestion does not duplicate entries."""
        with self._lock:
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
        with self._lock:
            if self._path_signature() != self._loaded_signature:
                self.load()
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
        with self._lock:
            return len(self.chunk_ids)

    def reset(self) -> None:
        with self._lock:
            self._clear()
            if self.path.exists():
                self.path.unlink()
