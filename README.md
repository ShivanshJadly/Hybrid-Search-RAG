# Hybrid-Search RAG over Internal Docs

Question answering over mixed-format internal documentation. Retrieves with
both dense (semantic) and sparse (keyword) search, fuses the two, reranks with
a cross-encoder, and answers with inline citations that are **verified rather
than trusted**.

Retrieval, embedding, chunking and reranking run entirely on local models;
generation and judging use Gemini's free tier. **95 tests**, all of which run
offline — every LLM-dependent path is also covered against a deterministic
stub, so no API key is needed to run the suite.

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
```

`torch` comes from the CPU index deliberately: the default index serves a ~2GB
CUDA build that nothing here needs. The embedding and reranker models are 33M
and 22M parameters — a second or two per query on CPU.

Add a free [Gemini key](https://aistudio.google.com/apikey) to `.env`
(copy `.env.example`), then confirm the wiring:

```bash
python -m rag check
```

`check` tests the things that actually fail, in order: key present, key valid,
configured model reachable *with that key*, and a live call returning text. If
the default model is not available to your account it prints the ones that are.

Then build the indexes:

```bash
python -m rag ingest-all --rebuild
```

## Use

```bash
python -m rag ask "how do I roll back a deploy"     # full answer pipeline
python -m rag search "CACHE_TTL_JITTER_PERCENT"     # raw dense vs sparse lists
python -m rag compare "how do I roll back"          # same query, 3 strategies
python -m rag eval                                  # golden set + metrics
python -m rag serve                                 # API + dashboard on :8000
python -m rag status                                # index state
python -m rag docs                                  # raw vs processed sizes
python -m pytest -q
```

Or `docker compose up --build`, which serves the API and dashboard on `:8000`
with the models baked into the image (see Limitations).

## Architecture

```
data/raw/*.md,.pdf,.html,.txt
   │
   ├─ load ────── clean text + anchors (headings, page breaks)
   ├─ chunk ───── fixed | header | semantic
   ├─ dedupe ──── exact (hash), then near (cosine >= 0.95)
   └─ index ───┬─ ChromaDB   (dense, 384-dim bge-small)
               └─ BM25       (sparse, identifier-preserving)
                     │
question ────────────┤
   ├─ dense top-10 + sparse top-10
   ├─ RRF fusion (0.7 / 0.3, k=60)
   ├─ cross-encoder rerank top-20 → top-5
   │
   ├─ grounded prompt → answer with [1] citations
   ├─ verify each claim against its cited chunk
   ├─ confidence = retrieval + coverage + citation accuracy
   └─ answer, or decline below threshold
                     │
   ├─ 52-question golden set → 7 metrics → strategy sweep
   └─ FastAPI + dashboard + Docker
```

The interesting modules: `retrieval.py` (RRF fusion), `verification.py`
(citation checking), `confidence.py` (composite scoring and the refusal scale),
`answering.py` (the single entry point the CLI, API and evaluator all share),
`chunking/` (three interchangeable strategies), `eval/` (golden set, metrics,
runner). Longer design notes, including what broke while building, are in
[`docs/`](docs/).

## Design decisions worth knowing

**Fusion is on rank, not score.** Cosine similarity is bounded 0–1; BM25 is
unbounded and corpus-dependent. Normalising them needs assumptions that break
whenever the corpus changes. Ranks are comparable by construction.

**Chunking strategies return character spans and nothing else.** A single
builder labels every span identically, so comparing strategies isolates the one
variable that should differ: where the cuts fall.

**The generator and the judge are configured separately.** A model grading its
own output is the weakest possible check, and grading is where a small model
underperforms most. Keeping them separate is what makes local generation with
free-tier judging a config change rather than a rewrite.

**An unparseable judge reply scores `PARTIAL`, never `SUPPORTED`.** Failing
open would silently inflate the headline citation-accuracy number.

**False refusal rate is tracked alongside refusal accuracy.** Without it,
"decline everything" scores 100% on refusal accuracy and faithfulness.

**Declining is a feature.** Below the confidence threshold the system reports
what it *did* find. A refusal is scored on its own scale — how sure we are that
refusing was right — because blending coverage and citation accuracy into a
number for an answer nobody sees rewards refusing while holding good material.

**Both indexes are written from the same post-dedup list in one pass**, and
every ingest asserts the counts match. Drift would silently degrade fusion with
nothing raising.

## Limitations

**The sample corpus is too easy.** Six files, and all three chunking strategies
hit 100% retrieval recall on it — so it cannot separate them. Point the pipeline
at a real documentation set before drawing any conclusion about which strategy
wins:

```bash
python -m rag ingest-all --rebuild /path/to/docs
```

**The full evaluation sweep has not been run against a live model.** A
5-question run (Gemini generating, a separate Gemini model judging) is recorded
in `storage/eval/`. The complete 52-question × 3-strategy sweep is ~450 model
calls, which exceeds a single day of free-tier quota. Use `--limit` to sample,
or `--no-grade` for a retrieval-only pass that skips the judge.

**The Docker image has not been built.** The Dockerfile and compose file are
written and internally consistent, but `docker compose up --build` has never
been run end to end.

**Semantic chunking is the expensive one.** It embeds every sentence, not every
chunk, so on a large corpus it dominates ingest cost. Its percentile threshold
is also per-document: a document uniformly about one thing still gets cut at its
own 95th percentile, because a percentile always finds something.

**Near-duplicate detection is an O(n·k) scan.** Fine into the tens of thousands
of chunks; beyond that it should become an approximate-neighbour lookup.
