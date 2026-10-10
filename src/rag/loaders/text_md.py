"""Plain-text and Markdown loader.

Markdown gets heading anchors from ATX (`#`) headings. Plain text has no
structure to extract, so it yields no heading anchors -- chunks from a .txt
file cite the filename only, which is the honest answer.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Anchor, LoadedDocument, make_doc_id
from .base import normalise

_ATX = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$", re.MULTILINE)


class TextLoader:
    extensions = (".txt",)
    media_type = "txt"

    def load(self, path: Path, source: str) -> LoadedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        return LoadedDocument(
            doc_id=make_doc_id(source),
            source=source,
            source_name=path.name,
            media_type=self.media_type,
            text=normalise(raw),
            raw_text=raw,
            anchors=[],
        )


class MarkdownLoader:
    extensions = (".md", ".markdown")
    media_type = "md"

    def load(self, path: Path, source: str) -> LoadedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = normalise(raw)
        anchors = [
            Anchor(
                offset=m.start(),
                kind="heading",
                value=m.group(2).strip(),
                level=len(m.group(1)),
            )
            for m in _ATX.finditer(text)
        ]
        return LoadedDocument(
            doc_id=make_doc_id(source),
            source=source,
            source_name=path.name,
            media_type=self.media_type,
            text=text,
            raw_text=raw,
            anchors=anchors,
        )
