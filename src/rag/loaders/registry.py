"""Extension -> loader dispatch."""

from __future__ import annotations

from pathlib import Path

from ..models import LoadedDocument
from .html import HtmlLoader
from .pdf import PdfLoader
from .text_md import MarkdownLoader, TextLoader

_LOADERS = [TextLoader(), MarkdownLoader(), HtmlLoader(), PdfLoader()]

_BY_EXT = {ext: ldr for ldr in _LOADERS for ext in ldr.extensions}

SUPPORTED_EXTENSIONS = tuple(sorted(_BY_EXT))


class UnsupportedFileType(Exception):
    pass


def load_file(path: Path, root: Path) -> LoadedDocument:
    ext = path.suffix.lower()
    loader = _BY_EXT.get(ext)
    if loader is None:
        raise UnsupportedFileType(f"No loader for '{ext}' ({path.name})")
    source = path.relative_to(root).as_posix()
    return loader.load(path, source)


def discover(root: Path) -> list[Path]:
    """Every ingestable file under root, in stable order."""
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _BY_EXT
    )
