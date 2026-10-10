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

    def save(self, doc: LoadedDocument) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
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
        path = self._path(doc.doc_id)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

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
