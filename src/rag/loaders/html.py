"""HTML loader.

Builds the text incrementally so that heading offsets are exact rather than
recovered by searching for the heading string afterwards (which breaks the
moment a heading's text repeats elsewhere in the page).
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from ..models import Anchor, LoadedDocument, make_doc_id
from .base import normalise

_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_BLOCKS = {"p", "li", "pre", "blockquote", "td", "th", "div", "section", "article"}
_DROP = {"script", "style", "nav", "footer", "noscript", "svg"}


class HtmlLoader:
    extensions = (".html", ".htm")
    media_type = "html"

    def load(self, path: Path, source: str) -> LoadedDocument:
        # The original markup is the raw form: it is the only way to see what
        # the parser dropped (scripts, nav, styling) when auditing extraction.
        raw = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "lxml")
        for tag in soup.find_all(_DROP):
            tag.decompose()

        parts: list[str] = []
        anchors: list[Anchor] = []
        cursor = 0

        body = soup.body or soup
        for el in body.find_all(list(_HEADINGS) + list(_BLOCKS)):
            # find_all is document-order; skip nested duplicates by only taking
            # elements whose own text is not already fully captured.
            content = el.get_text(" ", strip=True)
            if not content:
                continue
            if el.name in _HEADINGS:
                anchors.append(
                    Anchor(
                        offset=cursor,
                        kind="heading",
                        value=content,
                        level=_HEADINGS[el.name],
                    )
                )
            parts.append(content)
            cursor += len(content) + 2  # the "\n\n" join below

        text = "\n\n".join(parts)
        # normalise() may shift offsets; it is a no-op on text we built with
        # exactly "\n\n" separators, so anchors stay valid.
        text = text.strip()
        return LoadedDocument(
            doc_id=make_doc_id(source),
            source=source,
            source_name=path.name,
            media_type=self.media_type,
            text=text,
            raw_text=raw,
            anchors=[a for a in anchors if a.offset <= len(text)],
        )
