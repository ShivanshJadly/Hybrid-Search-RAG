"""Shared span-splitting helpers.

Both the header-aware and semantic strategies need the same fallback: given a
region of text that is too large for one chunk, cut it at the most natural
boundary available. Keeping that logic here means all three strategies enforce
the token ceiling identically, so a Phase 4 comparison is not quietly measuring
three different truncation behaviours.
"""

from __future__ import annotations

import re

from .base import Span, Tokenizer

# Ordered most-natural first. A paragraph break is a better cut than a line
# break, which is better than a sentence end, which is better than a space.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "; ", ", ", " ")

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[])|\n{2,}")


def sentence_spans(text: str, min_chars: int = 2) -> list[Span]:
    """Split text into sentence-ish spans, preserving character offsets.

    Deliberately simple. A full sentence tokeniser would handle abbreviations
    better, but the cost of an occasional bad split here is one slightly odd
    chunk boundary, not a wrong answer.

    Known limitation: a boundary requires terminal punctuation followed by an
    uppercase letter. That rule is what stops it cutting inside "v2.4 is
    required" or "e.g. this one", but it also means all-lowercase prose yields
    no boundaries at all. The semantic chunker then falls back to recursive
    splitting -- graceful, but no longer semantic. Pinned by
    test_sentence_splitter_needs_capitalised_sentences.
    """
    spans: list[Span] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.start()
        if end - cursor >= min_chars:
            spans.append(Span(cursor, end))
        cursor = match.end()
    if len(text) - cursor >= min_chars:
        spans.append(Span(cursor, len(text)))
    return spans or ([Span(0, len(text))] if text.strip() else [])


def split_to_budget(
    text: str,
    start: int,
    end: int,
    tokenizer: Tokenizer,
    max_tokens: int,
    _depth: int = 0,
) -> list[Span]:
    """Recursively cut [start, end) until every piece fits in max_tokens.

    Tries separators in order of naturalness, falling back to a hard token
    window if the text has no usable separator at all (a minified file, a long
    unbroken table).
    """
    region = text[start:end]
    if not region.strip():
        return []
    if tokenizer.count(region) <= max_tokens:
        return [Span(start, end)]

    for sep in _SEPARATORS[_depth:]:
        pieces = _split_on(text, start, end, sep)
        if len(pieces) < 2:
            continue
        out: list[Span] = []
        for piece in _merge_to_budget(text, pieces, tokenizer, max_tokens):
            if tokenizer.count(text[piece.start : piece.end]) <= max_tokens:
                out.append(piece)
            else:
                depth = _SEPARATORS.index(sep) + 1
                out.extend(
                    split_to_budget(
                        text, piece.start, piece.end, tokenizer, max_tokens, depth
                    )
                )
        if out:
            return out

    return _hard_window(text, start, end, tokenizer, max_tokens)


def _split_on(text: str, start: int, end: int, sep: str) -> list[Span]:
    """Split a region on a separator, keeping absolute offsets."""
    spans: list[Span] = []
    cursor = start
    idx = text.find(sep, cursor, end)
    while idx != -1:
        stop = idx + len(sep)
        if stop > cursor:
            spans.append(Span(cursor, stop))
        cursor = stop
        idx = text.find(sep, cursor, end)
    if end > cursor:
        spans.append(Span(cursor, end))
    return spans


def _merge_to_budget(
    text: str, pieces: list[Span], tokenizer: Tokenizer, max_tokens: int
) -> list[Span]:
    """Greedily recombine adjacent pieces so chunks fill the budget.

    Without this, splitting on "\\n\\n" turns a document into one chunk per
    paragraph, which throws away context the retriever needs.
    """
    out: list[Span] = []
    cur_start = pieces[0].start
    cur_end = pieces[0].start
    for piece in pieces:
        candidate = text[cur_start : piece.end]
        if cur_end > cur_start and tokenizer.count(candidate) > max_tokens:
            out.append(Span(cur_start, cur_end))
            cur_start = piece.start
        cur_end = piece.end
    if cur_end > cur_start:
        out.append(Span(cur_start, cur_end))
    return out


def _hard_window(
    text: str, start: int, end: int, tokenizer: Tokenizer, max_tokens: int
) -> list[Span]:
    """Last resort: cut on token boundaries with no regard for meaning."""
    offsets = tokenizer.offsets(text[start:end])
    if not offsets:
        return [Span(start, end)]
    out: list[Span] = []
    for i in range(0, len(offsets), max_tokens):
        window = offsets[i : i + max_tokens]
        s = start + window[0][0]
        e = start + window[-1][1]
        if e > s:
            out.append(Span(s, e))
    return out
