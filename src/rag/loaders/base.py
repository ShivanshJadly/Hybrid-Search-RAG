"""Shared text normalisation for the loaders.

A loader's job is to turn a file into clean text *and* record where the
structural landmarks sit inside that text. The anchors are the deliverable --
the plain text alone would make citations impossible later.
"""

from __future__ import annotations

import re

# Collapse 3+ blank lines, and trailing spaces, without disturbing offsets more
# than necessary. Applied *before* anchors are computed, never after.
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")


def normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WS.sub("\n", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()
