"""Document store: both the raw and the processed form of every source file.

Plan item 1 asks for raw *and* processed to be kept. Two reasons it earns its
place beyond literal compliance:

**Extraction auditing.** When a chunk reads badly, the question is always
whether the chunker cut poorly or the loader extracted poorly. Without the raw
form that is unanswerable without re-running the loader by hand. `list_documents`
surfaces the raw-vs-processed size gap, which is where bad extraction shows up
first.

One JSON file per document, keyed by doc_id. Small corpora only -- this is a
cache, not a database, and it stores text twice by design.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..models import Anchor, LoadedDocument


class DocumentStore:
    def __init__(self, persist_dir: Path) -> None:
        self.dir = Path(persist_dir)

    def _path(self, doc_id: str) -> Path:
        return self.dir / f"{doc_id}.json"

    def save(
        self,
        doc: LoadedDocument,
        *,
        content_hash: str | None = None,
        strategy: str | None = None,
    ) -> Path:
        if (content_hash is None) != (strategy is None):
            raise ValueError("content_hash and strategy must be provided together")
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(doc.doc_id)
        content_hashes: dict[str, str] = {}
        if strategy is not None and path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            content_hashes = previous.get("content_hashes", {})
        if strategy is not None:
            assert content_hash is not None
            content_hashes[strategy] = content_hash
        payload = {
            "doc_id": doc.doc_id,
            "source": doc.source,
            "source_name": doc.source_name,
            "media_type": doc.media_type,
            "raw_text": doc.raw_text,
            "text": doc.text,
            "anchors": [asdict(a) for a in doc.anchors],
            "extra": doc.extra,
        }
        if content_hashes:
            payload["content_hashes"] = content_hashes
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def content_hash(self, doc_id: str, strategy: str) -> str | None:
        path = self._path(doc_id)
        if not path.exists():
            return None
        blob = json.loads(path.read_text(encoding="utf-8"))
        return blob.get("content_hashes", {}).get(strategy)

    def invalidate_fingerprint(self, strategy: str) -> None:
        if not self.dir.exists():
            return
        for path in self.dir.glob("*.json"):
            blob = json.loads(path.read_text(encoding="utf-8"))
            content_hashes = blob.get("content_hashes", {})
            if strategy not in content_hashes:
                continue
            del content_hashes[strategy]
            if content_hashes:
                blob["content_hashes"] = content_hashes
            else:
                blob.pop("content_hashes", None)
            path.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")

    def delete(self, doc_id: str) -> bool:
        path = self._path(doc_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def load(self, doc_id: str) -> LoadedDocument | None:
        path = self._path(doc_id)
        if not path.exists():
            return None
        blob = json.loads(path.read_text(encoding="utf-8"))
        return LoadedDocument(
            doc_id=blob["doc_id"],
            source=blob["source"],
            source_name=blob["source_name"],
            media_type=blob["media_type"],
            text=blob["text"],
            raw_text=blob.get("raw_text", ""),
            anchors=[Anchor(**a) for a in blob.get("anchors", [])],
            extra=blob.get("extra", {}),
        )

    def list_documents(self) -> list[dict]:
        """Summary rows for the CLI. Does not load the text bodies."""
        if not self.dir.exists():
            return []
        rows = []
        for path in sorted(self.dir.glob("*.json")):
            blob = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "doc_id": blob["doc_id"],
                    "source": blob["source"],
                    "media_type": blob["media_type"],
                    "raw_chars": len(blob.get("raw_text", "")),
                    "text_chars": len(blob["text"]),
                    "anchors": len(blob.get("anchors", [])),
                }
            )
        return rows

    def count(self) -> int:
        return len(list(self.dir.glob("*.json"))) if self.dir.exists() else 0
