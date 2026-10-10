# Phase 1 — Ingestion & Chunking

**Plan reference:** Notion, Phase 1 (Day 1-3). **Status: complete.**

## What Phase 1 built

```
data/raw/*  →  load  →  chunk  →  exact-dedup  →  embed  →  near-dedup  ─┬→  ChromaDB (dense)
                                                                        └→  BM25     (sparse)
```

| Piece | Status |
|---|---|
| Loaders: `.md` `.txt` `.html` `.htm` `.pdf` | done |
| Raw + processed text kept per document | done |
| Heading + page provenance | done |
| Fixed-size chunking (token windows, overlap) | done |
| Header-aware chunking (structure-aware) | done |
| Semantic chunking (embedding-distance boundaries) | done |
| Local embeddings (`bge-small-en-v1.5`, 384-dim) | done |
| ChromaDB vector index | done |
| BM25 sparse index | done |
| Exact-duplicate removal (content hash) | done |
| Near-duplicate removal (cosine ≥ 0.95) | done |
| `LLMClient` interface | done (no implementations yet) |

Current sample corpus (6 files):

| strategy | chunks | largest (tokens) |
|---|---|---|
| fixed | 12 | 400 |
| header | 13 | 393 |
| semantic | 18 | 378 |

## Design decisions worth knowing

**Loaders emit text *plus anchors*.** An anchor marks where a heading or page
break sits in the cleaned text, so any character offset resolves back to a
heading breadcrumb and a page number. This is the entire basis of Phase 3
citations — metadata not captured at load time cannot be recovered without
re-ingesting.

**Chunking strategies receive the document and return character spans, nothing
else.** They never build metadata. A single builder (`chunking/builder.py`)
labels every span identically regardless of strategy, so Phase 4's comparison
isolates the one variable that should differ: where the cuts fall.

**All three strategies enforce the token ceiling through the same code**
(`chunking/splitting.py`). Otherwise a strategy comparison would quietly be
measuring three different truncation behaviours alongside three chunking
behaviours.

**Both indexes are written from the same post-dedup list in one pass**, and
every ingest asserts the counts match. Drift between Chroma and BM25 would
silently degrade Phase 2 fusion with nothing raising.

**Chunk IDs are content-derived**, so re-ingesting an unchanged corpus is
idempotent rather than duplicating every chunk.

**Chunk size is validated against the embedding model's token limit at
startup.** `bge-small-en-v1.5` truncates past 512 tokens without warning — an
oversized chunk still embeds, still stores and still retrieves, just missing
its tail. `validate_chunk_size()` refuses to run rather than allow that.

**BM25 tokenisation preserves identifiers.** `DATABASE_POOL_SIZE`,
`ERR_DB_5031` and `canary-policy.yaml` survive as single tokens. Catching exact
identifiers that semantic search misses is the entire reason the sparse index
exists, and a naive word split would defeat it.

**bge queries get an asymmetric prefix, passages do not.** The model was
trained that way; omitting the prefix on the query path measurably hurts
retrieval and is easy to forget.

**Both the raw and processed form of every document are kept.** `text` is what
everything downstream uses; `raw_text` is what the extractor produced before
normalisation. It earns its place twice over: re-chunking under a new strategy
need not re-parse every PDF, and when a chunk reads badly it answers the
question of whether the chunker cut poorly or the loader extracted poorly. The
gap is visible in `python -m rag docs` -- HTML loses ~14% to markup and nav,
Markdown loses one trailing newline. A Markdown file showing a large gap would
mean the loader is eating content.

## Things found while building

**Boundary snapping overshot the token budget.** Completing a partial word at
each chunk edge added subword tokens the window had not budgeted for — 402
tokens against a 400 budget on the real corpus. Harmless at 400, but at a
chunk size near the model's 512 ceiling it would have caused silent
truncation. Fixed by keeping the tidier boundaries only when they still fit.
Pinned by `test_fixed_chunker_does_not_overshoot_with_subword_tokens`, which
needs a subword-style tokenizer to reproduce — a word-level fake cannot catch
it.

**Cosine ≥ 0.95 only catches near-verbatim chunks.** A deliberately copied
section between two documents scored **0.9258** and was correctly kept,
because the containing chunk also held surrounding text that differed. A
one-character-different document scored **1.000** and was dropped. The
threshold is doing what it says; just do not expect it to catch paraphrase.

**The sentence splitter requires a capital letter after terminal
punctuation.** That is what stops it cutting inside `v2.4 is required` or
`e.g. this`, but all-lowercase prose yields no boundaries at all and semantic
chunking silently degrades to recursive splitting. Graceful, but no longer
semantic. Pinned by `test_sentence_splitter_needs_capitalised_sentences`.

## Known limitations

- **A chunk spanning a page boundary cites its starting page.** A chunk
  beginning on p.1 and running into p.2 is cited as p.1. Correct as designed;
  if Phase 3 needs finer citations, store a page range instead.
- **Semantic chunking always finds boundaries**, because a percentile always
  returns a value. A document uniformly about one topic still gets cut at its
  95th-percentile distances. Inherent to the method, and exactly the kind of
  thing Phase 4 should surface.
- **Header merging trades attribution for chunk size.** Adjacent short
  sections are merged while they fit, and the merged chunk cites the *first*
  section's heading. Set `header_merge_small=False` to measure the tradeoff.
- **Semantic chunking is much more expensive** — every sentence is embedded,
  not just every chunk. Dominant ingest cost on a large corpus.
- **Chunk size and overlap (400/60) are guesses.** Phase 4 replaces them with
  measured values. Do not tune them by eye before then.
- **Near-duplicate detection is an O(n·k) scan.** Fine into the tens of
  thousands of chunks; beyond that it needs an approximate-neighbour index.
- **PDF extraction is imperfect** and we accept that. Image-only PDFs yield no
  text and are reported as "no extractable text" rather than failing.

## Layout

```
src/rag/
  models.py            data structures, AnchorResolver, id generation
  config.py            all tunables, chunk-size validation
  embeddings.py        local embedder + tokenizer surface
  dedup.py             near-duplicate removal
  pipeline.py          ingest / search orchestration, index-sync assertion
  cli.py               typer CLI
  loaders/             per-format loaders + dispatch
  chunking/
    base.py            Span / ChunkingStrategy / Tokenizer contracts
    splitting.py       shared sentence + budget splitting
    fixed.py           baseline: token windows with overlap
    header.py          structure-aware: cut on the document's headings
    semantic.py        topic boundaries via embedding distance
    builder.py         the single place a span becomes a labelled Chunk
    registry.py        strategy lookup
  stores/
    vector.py          Chroma (dense)
    bm25.py            rank_bm25 (sparse)
    documents.py       raw + processed text per document
  llm/base.py          the LLMClient seam (Appendix A.6)
scripts/
  make_sample_pdf.py   regenerates the PDF fixture (needs reportlab)
tests/
  test_phase1.py       47 tests
```
