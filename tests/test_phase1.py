"""Phase 1 tests.

Weighted towards the things that would sink later phases if they went wrong:
provenance metadata being captured correctly, every strategy respecting the
token ceiling identically, and the two indexes staying in step.

The real embedding model is never loaded here -- a fake tokenizer and a
deterministic fake embedder keep these tests fast and offline.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from rag.chunking.builder import build_chunks
from rag.chunking.fixed import FixedSizeChunker
from rag.chunking.header import HeaderChunker
from rag.chunking.registry import STRATEGIES, UnknownStrategy, get_strategy
from rag.chunking.semantic import SemanticChunker
from rag.chunking.splitting import sentence_spans, split_to_budget
from rag.config import Settings, validate_chunk_size
from rag.dedup import drop_near_duplicates
from rag.loaders.registry import discover, load_file
from rag.models import (
    Anchor,
    AnchorResolver,
    Chunk,
    ChunkMetadata,
    LoadedDocument,
    content_fingerprint,
    make_chunk_id,
)
from rag.stores.bm25 import tokenize
from rag.stores.documents import DocumentStore

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
needs_corpus = pytest.mark.skipif(not DATA.exists(), reason="sample corpus absent")


_WORD = re.compile(r"\S+")


class WordTokenizer:
    """One token per whitespace-separated word.

    offsets() and count() must agree exactly -- an earlier version split on
    " " in one and on all whitespace in the other, which made the tokenizer
    disagree with itself across newlines and produced phantom budget
    overflows in these tests.
    """

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in _WORD.finditer(text)]

    def count(self, text: str) -> int:
        return len(_WORD.findall(text))


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic pseudo-embeddings.

    Texts sharing vocabulary land near each other, which is enough for the
    semantic chunker's boundary logic to be exercised without a real model.
    """
    dim = 16
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for word in t.lower().split():
            out[i, hash(word) % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def doc_of(text: str, anchors: list[Anchor] | None = None) -> LoadedDocument:
    return LoadedDocument(
        doc_id="d",
        source="a.md",
        source_name="a.md",
        media_type="md",
        text=text,
        anchors=anchors or [],
    )


def chunk_of(text: str, ordinal: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"c{ordinal}",
        doc_id="d",
        text=text,
        meta=ChunkMetadata(
            source="a.md",
            source_name="a.md",
            media_type="md",
            strategy="fixed",
            ordinal=ordinal,
            char_start=0,
            char_end=len(text),
            n_tokens=len(text.split()),
        ),
    )


# --------------------------------------------------------------- provenance


def test_heading_breadcrumb_walks_up_levels():
    doc = doc_of(
        "x" * 400,
        [
            Anchor(0, "heading", "Guide", 1),
            Anchor(100, "heading", "Failures", 2),
            Anchor(200, "heading", "Database", 3),
        ],
    )
    r = AnchorResolver(doc)
    assert r.heading_at(50) == ("Guide", "Guide")
    assert r.heading_at(150) == ("Failures", "Guide > Failures")
    assert r.heading_at(250) == ("Database", "Guide > Failures > Database")


def test_offset_before_first_heading_has_no_breadcrumb():
    doc = doc_of("text here", [Anchor(50, "heading", "H", 1)])
    assert AnchorResolver(doc).heading_at(10) == ("", "")


def test_page_lookup_returns_containing_page():
    doc = doc_of(
        "x" * 300,
        [Anchor(0, "page", "1"), Anchor(100, "page", "2"), Anchor(200, "page", "3")],
    )
    r = AnchorResolver(doc)
    assert r.page_at(0) == 1
    assert r.page_at(150) == 2
    assert r.page_at(299) == 3


def test_chunk_ids_are_deterministic_and_content_sensitive():
    a = make_chunk_id("doc", "fixed", 0, "hello world")
    b = make_chunk_id("doc", "fixed", 0, "hello world")
    c = make_chunk_id("doc", "fixed", 0, "hello worlds")
    assert a == b, "re-ingesting unchanged content must not mint new ids"
    assert a != c


def test_ids_differ_across_strategies_for_identical_text():
    """Strategies live in separate collections but share an id namespace in
    logs and reports; identical text must not collide across them."""
    assert make_chunk_id("doc", "fixed", 0, "same") != make_chunk_id(
        "doc", "header", 0, "same"
    )


def test_fingerprint_ignores_case_and_whitespace():
    assert content_fingerprint("Hello   World\n") == content_fingerprint("hello world")


# ------------------------------------------------------- token-budget rules


@pytest.mark.parametrize("strategy_name", STRATEGIES)
def test_every_strategy_respects_the_token_budget(strategy_name):
    """The quietest failure mode in the pipeline: an oversized chunk still
    embeds and still retrieves, just with its tail silently discarded. No
    strategy is allowed to produce one."""
    tok = WordTokenizer()
    settings = Settings(chunk_tokens=40, chunk_overlap_tokens=8)
    strategy = get_strategy(strategy_name, tok, settings, embed_fn=fake_embed)

    body = ". ".join(
        f"Sentence {i} about topic {i // 7} with some filler words" for i in range(60)
    )
    text = f"# Title\n\n{body}\n\n## Section Two\n\n{body}"
    doc = doc_of(text, [Anchor(0, "heading", "Title", 1), Anchor(len(body) + 11, "heading", "Section Two", 2)])

    spans = strategy.spans(doc)
    assert spans, f"{strategy_name} produced no spans"
    for s in spans:
        assert tok.count(text[s.start : s.end]) <= 40, f"{strategy_name} overflowed"


@pytest.mark.parametrize("strategy_name", STRATEGIES)
def test_every_strategy_handles_empty_and_tiny_documents(strategy_name):
    tok = WordTokenizer()
    settings = Settings()
    strategy = get_strategy(strategy_name, tok, settings, embed_fn=fake_embed)
    assert strategy.spans(doc_of("   ")) == []
    assert len(strategy.spans(doc_of("One short line of text."))) == 1


@pytest.mark.parametrize("strategy_name", STRATEGIES)
def test_spans_are_ordered_and_within_bounds(strategy_name):
    tok = WordTokenizer()
    settings = Settings(chunk_tokens=30, chunk_overlap_tokens=5)
    strategy = get_strategy(strategy_name, tok, settings, embed_fn=fake_embed)
    text = "# H\n\n" + ". ".join(f"word{i} filler text here" for i in range(80))
    doc = doc_of(text, [Anchor(0, "heading", "H", 1)])

    spans = strategy.spans(doc)
    for s in spans:
        assert 0 <= s.start < s.end <= len(text)
    for prev, nxt in zip(spans, spans[1:]):
        assert nxt.start >= prev.start


# ----------------------------------------------------------- fixed strategy


def test_fixed_chunks_overlap_and_advance():
    tok = WordTokenizer()
    text = " ".join(f"word{i}" for i in range(300))
    spans = FixedSizeChunker(tok, chunk_tokens=50, overlap_tokens=10).spans(doc_of(text))
    assert len(spans) > 1
    for prev, nxt in zip(spans, spans[1:]):
        assert nxt.start > prev.start, "spans must advance"
        assert nxt.start < prev.end, "consecutive spans must overlap"


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        FixedSizeChunker(WordTokenizer(), chunk_tokens=50, overlap_tokens=50)


class SubwordTokenizer:
    """Cuts every 3 characters, mimicking how a real WordPiece tokenizer
    splits mid-word. WordTokenizer cannot catch boundary-snapping overshoot
    because its tokens are already whole words."""

    STRIDE = 3

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [
            (i, min(i + self.STRIDE, len(text)))
            for i in range(0, len(text), self.STRIDE)
            if text[i : i + self.STRIDE].strip()
        ]

    def count(self, text: str) -> int:
        return len(self.offsets(text))


def test_fixed_chunker_does_not_overshoot_with_subword_tokens():
    """Regression: snapping to word boundaries completed a partial word at
    each edge and pushed chunks past the budget -- 402 tokens against a 400
    budget on the real corpus. Harmless at 400, silent truncation near the
    model's 512 ceiling."""
    tok = SubwordTokenizer()
    text = " ".join(f"antidisestablishmentarianism{i}" for i in range(60))
    for span in FixedSizeChunker(tok, chunk_tokens=40, overlap_tokens=8).spans(
        doc_of(text)
    ):
        assert tok.count(text[span.start : span.end]) <= 40


# ---------------------------------------------------------- header strategy


def test_header_chunker_cuts_on_headings():
    tok = WordTokenizer()
    body_a = " ".join(f"alpha{i}" for i in range(40))
    body_b = " ".join(f"beta{i}" for i in range(40))
    text = f"# One\n\n{body_a}\n\n# Two\n\n{body_b}"
    anchors = [
        Anchor(0, "heading", "One", 1),
        Anchor(text.index("# Two"), "heading", "Two", 1),
    ]
    spans = HeaderChunker(tok, chunk_tokens=200, merge_small=False).spans(
        doc_of(text, anchors)
    )
    assert len(spans) == 2
    assert "alpha0" in text[spans[0].start : spans[0].end]
    assert "beta0" in text[spans[1].start : spans[1].end]
    assert "beta0" not in text[spans[0].start : spans[0].end]


def test_header_chunker_keeps_text_before_the_first_heading():
    tok = WordTokenizer()
    text = "Intro paragraph that precedes any heading at all.\n\n# One\n\nBody here."
    anchors = [Anchor(text.index("# One"), "heading", "One", 1)]
    spans = HeaderChunker(tok, chunk_tokens=200, merge_small=False).spans(
        doc_of(text, anchors)
    )
    assert "Intro paragraph" in text[spans[0].start : spans[0].end]


def test_header_chunker_merges_small_adjacent_sections():
    tok = WordTokenizer()
    text = "# A\n\nshort one\n\n# B\n\nshort two\n\n# C\n\nshort three"
    anchors = [
        Anchor(0, "heading", "A", 1),
        Anchor(text.index("# B"), "heading", "B", 1),
        Anchor(text.index("# C"), "heading", "C", 1),
    ]
    merged = HeaderChunker(tok, chunk_tokens=200, merge_small=True).spans(
        doc_of(text, anchors)
    )
    split = HeaderChunker(tok, chunk_tokens=200, merge_small=False).spans(
        doc_of(text, anchors)
    )
    assert len(merged) < len(split)


def test_header_chunker_falls_back_when_no_headings():
    tok = WordTokenizer()
    text = " ".join(f"word{i}" for i in range(200))
    spans = HeaderChunker(tok, chunk_tokens=50).spans(doc_of(text))
    assert len(spans) > 1
    for s in spans:
        assert tok.count(text[s.start : s.end]) <= 50


# -------------------------------------------------------- semantic strategy


def test_semantic_chunker_splits_at_a_topic_change():
    tok = WordTokenizer()
    # Two blocks with disjoint vocabulary; the boundary between them should be
    # the largest embedding distance in the document. Sentences must be
    # capitalised -- see test_sentence_splitter_needs_capitalised_sentences.
    a = " ".join("Alpha beta gamma delta epsilon." for _ in range(6))
    b = " ".join("Zulu yankee xray whiskey victor." for _ in range(6))
    text = f"{a} {b}"
    chunker = SemanticChunker(tok, fake_embed, chunk_tokens=500, percentile=90.0)
    spans = chunker.spans(doc_of(text))
    assert len(spans) > 1, "expected at least one topic boundary"


def test_sentence_splitter_needs_capitalised_sentences():
    """A documented limitation, pinned so it cannot regress silently.

    The splitter requires a capital letter after terminal punctuation, which
    is what stops it cutting inside 'v2.4 is required' or 'e.g. this'. The
    cost is that all-lowercase prose yields no sentence boundaries at all --
    the semantic chunker then falls back to recursive splitting, which is
    graceful but is not semantic chunking.
    """
    assert len(sentence_spans("alpha one. beta two. gamma three.")) == 1
    assert len(sentence_spans("Alpha one. Beta two. Gamma three.")) == 3
    # The reason the rule exists:
    assert len(sentence_spans("Upgrade to v2.4 is required today.")) == 1


def test_semantic_chunker_falls_back_on_short_documents():
    tok = WordTokenizer()
    chunker = SemanticChunker(tok, fake_embed, chunk_tokens=500, min_sentences=4)
    spans = chunker.spans(doc_of("One. Two."))
    assert len(spans) == 1


# ---------------------------------------------------------------- splitting


def test_sentence_spans_cover_text_without_overlap():
    text = "First one. Second one! Third one? Fourth."
    spans = sentence_spans(text)
    assert len(spans) >= 3
    for prev, nxt in zip(spans, spans[1:]):
        assert prev.end <= nxt.start


def test_split_to_budget_handles_text_with_no_separators():
    tok = WordTokenizer()
    text = "x" * 5000  # one enormous token, no spaces anywhere
    spans = split_to_budget(text, 0, len(text), tok, max_tokens=10)
    assert spans
    for s in spans:
        assert s.end > s.start


# ---------------------------------------------------------------- dedup


def test_near_duplicates_are_dropped_keeping_the_first():
    chunks = [chunk_of("alpha beta gamma", 0), chunk_of("alpha beta gamma delta", 1)]
    vectors = np.array([[1.0, 0.0], [0.999, 0.0447]], dtype=np.float32)
    result = drop_near_duplicates(chunks, vectors, threshold=0.95)
    assert result.removed == 1
    assert result.chunks[0].meta.ordinal == 0, "first occurrence must win"
    assert len(result.vectors) == 1


def test_distinct_chunks_are_kept():
    chunks = [chunk_of("alpha", 0), chunk_of("zulu", 1)]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert drop_near_duplicates(chunks, vectors, threshold=0.95).removed == 0


def test_dedup_keeps_chunks_and_vectors_aligned():
    chunks = [chunk_of(f"text {i}", i) for i in range(5)]
    vectors = np.eye(5, dtype=np.float32)
    vectors[3] = vectors[0]  # make index 3 a duplicate of index 0
    result = drop_near_duplicates(chunks, vectors, threshold=0.95)
    assert len(result.chunks) == len(result.vectors)
    assert [c.meta.ordinal for c in result.chunks] == [0, 1, 2, 4]


def test_dedup_rejects_misaligned_input():
    with pytest.raises(ValueError, match="not aligned"):
        drop_near_duplicates([chunk_of("a")], np.eye(2, dtype=np.float32))


# ------------------------------------------------------------------ config


def test_oversized_chunk_config_is_rejected():
    with pytest.raises(ValueError, match="silently truncated"):
        validate_chunk_size(chunk_tokens=600, model_max_tokens=512)


def test_semantic_strategy_requires_an_embedder():
    with pytest.raises(UnknownStrategy, match="embed_fn"):
        get_strategy("semantic", WordTokenizer(), Settings())


def test_unknown_strategy_lists_the_valid_ones():
    with pytest.raises(UnknownStrategy, match="fixed"):
        get_strategy("nonsense", WordTokenizer(), Settings())


# ------------------------------------------------------------------ loaders


@needs_corpus
def test_markdown_headings_point_at_real_headings():
    doc = load_file(DATA / "deploy-guide.md", DATA)
    assert doc.media_type == "md"
    headings = doc.headings()
    assert headings, "markdown loader found no headings"
    for a in headings:
        assert doc.text[a.offset : a.offset + 200].lstrip("#").strip().startswith(a.value)


@needs_corpus
def test_pdf_pages_are_sequential_from_one():
    doc = load_file(DATA / "data-retention-policy.pdf", DATA)
    pages = [int(a.value) for a in doc.pages()]
    assert pages == sorted(pages)
    assert pages[0] == 1


@needs_corpus
def test_html_drops_nav_and_keeps_headings():
    doc = load_file(DATA / "onboarding.html", DATA)
    assert "Engineering Onboarding" in doc.text
    assert doc.headings()
    for a in doc.headings():
        assert a.offset <= len(doc.text)


@needs_corpus
def test_discover_is_stable_and_finds_every_type():
    found = discover(DATA)
    assert found == sorted(found)
    assert {p.suffix for p in found} >= {".md", ".pdf", ".html", ".txt"}


# ---------------------------------------------------------------- labelling


@needs_corpus
@pytest.mark.parametrize("strategy_name", STRATEGIES)
def test_every_chunk_carries_a_usable_citation(strategy_name):
    doc = load_file(DATA / "deploy-guide.md", DATA)
    tok = WordTokenizer()
    settings = Settings(chunk_tokens=120, chunk_overlap_tokens=20)
    strategy = get_strategy(strategy_name, tok, settings, embed_fn=fake_embed)

    chunks = build_chunks(doc, strategy, tok)
    assert chunks
    for c in chunks:
        assert c.meta.source_name == "deploy-guide.md"
        assert c.meta.strategy == strategy_name
        assert c.meta.char_end > c.meta.char_start
        assert c.citation().startswith("deploy-guide.md")


# ------------------------------------------------------------ document store


def test_document_store_round_trips_raw_and_processed(tmp_path):
    store = DocumentStore(tmp_path)
    doc = LoadedDocument(
        doc_id="abc123",
        source="sub/a.md",
        source_name="a.md",
        media_type="md",
        text="Processed body.",
        raw_text="# Raw\n\n\n\nProcessed body.",
        anchors=[Anchor(0, "heading", "Raw", 1)],
        extra={"n_pages": 0},
    )
    store.save(doc)

    back = store.load("abc123")
    assert back is not None
    assert back.text == doc.text
    assert back.raw_text == doc.raw_text, "raw form must survive the round trip"
    assert back.anchors == doc.anchors
    assert back.extra == doc.extra


def test_document_store_returns_none_for_unknown_id(tmp_path):
    assert DocumentStore(tmp_path).load("nope") is None


def test_document_store_lists_without_loading_bodies(tmp_path):
    store = DocumentStore(tmp_path)
    for i in range(3):
        store.save(
            LoadedDocument(
                doc_id=f"id{i}",
                source=f"{i}.md",
                source_name=f"{i}.md",
                media_type="md",
                text="body",
                raw_text="raw body",
            )
        )
    rows = store.list_documents()
    assert store.count() == 3
    assert {r["doc_id"] for r in rows} == {"id0", "id1", "id2"}
    assert all(r["raw_chars"] == 8 and r["text_chars"] == 4 for r in rows)


@needs_corpus
def test_loaders_all_populate_raw_text():
    """Plan item 1: keep raw *and* processed. Every loader must do it, not
    just the text ones."""
    for name in (
        "deploy-guide.md",
        "onboarding.html",
        "changelog.txt",
        "data-retention-policy.pdf",
    ):
        doc = load_file(DATA / name, DATA)
        assert doc.raw_text, f"{name} kept no raw text"
        assert doc.text, f"{name} produced no processed text"


@needs_corpus
def test_html_raw_keeps_markup_that_processing_removes():
    doc = load_file(DATA / "onboarding.html", DATA)
    assert "<nav>" in doc.raw_text, "raw HTML must retain the original markup"
    assert "<nav>" not in doc.text
    assert "Home" not in doc.text, "nav content should be stripped from text"


# -------------------------------------------------------------------- BM25


def test_tokenizer_keeps_identifiers_intact():
    """The entire reason BM25 is in this system: exact identifiers that
    semantic search is bad at."""
    assert "database_pool_size" in tokenize("Set DATABASE_POOL_SIZE to 20")
    assert "err_db_5031" in tokenize("fails with ERR_DB_5031 in logs")
    assert "canary-policy.yaml" in tokenize("see canary-policy.yaml at root")
