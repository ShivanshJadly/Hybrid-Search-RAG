"""PDF loader.

PDF text extraction is imperfect and we accept that -- perfect extraction is
not what this project demonstrates. What we do care about is page numbers,
because "p.7" is half of a usable citation.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from ..models import Anchor, LoadedDocument, make_doc_id


class PdfLoader:
    extensions = (".pdf",)
    media_type = "pdf"

    def load(self, path: Path, source: str) -> LoadedDocument:
        reader = PdfReader(str(path))
        parts: list[str] = []
        raw_parts: list[str] = []
        anchors: list[Anchor] = []
        cursor = 0

        for i, page in enumerate(reader.pages, start=1):
            try:
                extracted = page.extract_text() or ""
            except Exception:
                extracted = ""
            # Keep the un-normalised extraction: PDF artefacts (ligatures,
            # stray hyphenation, column bleed) are visible there and gone
            # after stripping.
            raw_parts.append(f"[page {i}]\n{extracted}")
            content = extracted.strip()
            if not content:
                continue
            anchors.append(
                Anchor(offset=cursor, kind="page", value=str(i), level=0)
            )
            parts.append(content)
            cursor += len(content) + 2

        return LoadedDocument(
            doc_id=make_doc_id(source),
            source=source,
            source_name=path.name,
            media_type=self.media_type,
            text="\n\n".join(parts),
            raw_text="\n\n".join(raw_parts),
            anchors=anchors,
            extra={"n_pages": len(reader.pages)},
        )
