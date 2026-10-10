# Hybrid-Search RAG over Internal Docs

Question answering over internal documentation. Retrieves with both dense
(semantic) and sparse (keyword) search, fuses the two, reranks with a
cross-encoder, and answers with inline citations that are **verified rather
than trusted**.

**Zero cost.** Retrieval, embedding, chunking and reranking run entirely on
local models. Generation and judging use Gemini's free tier. See the Notion
plan, Appendix A, for the free/paid split and its rationale.

## Status

| Phase | Scope | Status |
|---|---|---|
| [1 — Ingestion & Chunking](docs/phase-1-ingestion.md) | Loaders, 3 chunking strategies, dual indexes, dedup | **complete** |
| [2 — Hybrid Retrieval](docs/phase-2-retrieval.md) | RRF fusion, cross-encoder reranking | **complete** |
| [3 — Generation & Citation](docs/phase-3-generation.md) | Grounded answers, citation verification, confidence | **complete** |
| [4 — Evaluation](docs/phase-4-evaluation.md) | 52-question golden set, 7 metrics, strategy sweep | **complete — full 52×3 sweep not yet run** |
| [5 — API & Dashboard](docs/phase-5-api.md) | FastAPI, dashboard, Docker | **complete — image unbuilt** |
| 6 — Portfolio polish | Demo video, case study | not started |

**95 tests pass.** Every LLM-dependent path is also verified against a stub, so
the suite runs offline with no key. A 5-question real-model run (Gemini
generating, a separate Gemini model judging) is in `storage/eval/`; the full
52-question × 3-strategy sweep is ~450 model calls and has not been run — see
[What still needs you](#what-still-needs-you).

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -e .
python -m rag ingest-all --rebuild
```

Then put your Gemini key in `.env` (see [What still needs you](#what-still-needs-you))
and run `python -m rag check`.

`torch` comes from the CPU index deliberately: the default index serves a ~2GB
CUDA build this machine cannot use, because PyTorch has no ROCm build for
Windows. The embedding and reranker models are small enough that CPU is fine —
33M and 22M parameters, a second or two per query.

## Use

```bash
python -m rag ingest-all --rebuild                  # index under all 3 strategies
python -m rag ask "how do I roll back a deploy"     # full answer pipeline
python -m rag search "CACHE_TTL_JITTER_PERCENT"     # raw dense vs sparse lists
python -m rag compare "how do I roll back"          # same query, 3 strategies
python -m rag eval                                  # golden set + metrics
python -m rag serve                                 # API + dashboard on :8000
python -m rag check                                 # are the backends reachable?
python -m rag status                                # index state
python -m rag docs                                  # raw vs processed sizes
python -m pytest -q
```

## Architecture

```
data/raw/*.md,.pdf,.html,.txt
   │
   ├─ load ────── clean text + anchors (headings, page breaks)   ── Phase 1
   ├─ chunk ───── fixed | header | semantic
   ├─ dedupe ──── exact (hash), then near (cosine ≥ 0.95)
   └─ index ───┬─ ChromaDB   (dense, 384-dim bge-small)
               └─ BM25       (sparse, identifier-preserving)
                     │
question ────────────┤                                            ── Phase 2
   ├─ dense top-10 + sparse top-10
   ├─ RRF fusion (0.7 / 0.3, k=60)
   ├─ cross-encoder rerank top-20 → top-5
   │
   ├─ grounded prompt → answer with [1] citations                 ── Phase 3
   ├─ verify each claim against its cited chunk
   ├─ confidence = retrieval + coverage + citation accuracy
   └─ answer, or decline below threshold
                     │
   ├─ golden set (52 q) → 7 metrics → strategy sweep              ── Phase 4
   └─ FastAPI + dashboard + Docker                                ── Phase 5
```

## Design decisions worth knowing

**Fusion is on rank, not score.** Cosine similarity is bounded 0–1; BM25 is
unbounded and corpus-dependent. Normalising them needs assumptions that break
when the corpus changes. Ranks are comparable by construction.

**Chunking strategies return character spans and nothing else.** A single
builder labels every span identically, so Phase 4's comparison isolates the one
variable that should differ: where the cuts fall.

**The generator and the judge are configured separately.** A model grading its
own output is the weakest possible check, and grading is exactly where a small
local model underperforms. This is what makes the plan's free arrangement —
local generation, free-tier judging — a config change.

**An unparseable judge reply scores `PARTIAL`, never `SUPPORTED`.** Failing
open would silently inflate the one number the portfolio claim rests on.

**False refusal rate is tracked alongside refusal accuracy.** Without it,
"decline everything" scores 100% on refusal accuracy and faithfulness.

**Declining is a feature.** Below the confidence threshold the system reports
what it *did* find. An honest refusal is scored as a correct non-answer, not a
bad answer — otherwise the metrics push the system toward guessing.

**Both indexes are written from the same post-dedup list in one pass**, and
every ingest asserts the counts match. Drift would silently degrade fusion with
nothing raising.

## What still needs you

Three things I could not do from here.

**1. Add your Gemini API key.** Everything else is wired; this is the one
value the project cannot supply itself.

Get a free key at <https://aistudio.google.com/apikey>, then open `.env` in the
project root and paste it in:

```
GEMINI_API_KEY=AIzaSy...
```

Then verify:

```bash
python -m rag check
```

`check` tests the things that actually fail, in order: key present, key valid,
configured model reachable *with that key*, and a live call returning text. If
the default model is not available to your account it prints the ones that are
— paste one into `RAG_GEMINI_MODEL` in `.env`.

Gemini is set as both the generator and the judge (`RAG_LLM_BACKEND` /
`RAG_JUDGE_BACKEND`). They are separate settings, so you can point the judge at
a different model later without touching generation.

**2. Run the full evaluation.** A 5-question real-model run is recorded in
`storage/eval/`; the headline strategy comparison needs the whole set:

```bash
python -m rag eval --limit 10      # start small, watch the quota
python -m rag eval                 # full 52 questions, all 3 strategies
```

Free-tier quotas are account-specific and rate limits bite: a full sweep is
~450 model calls. Start with `--limit`, and use `--no-grade` for a fast
retrieval-only pass that costs nothing.

**3. Build the Docker image once.** The Dockerfile and compose file are written
and internally consistent but have never been run.

```bash
docker compose up --build
```

Beyond that: **the sample corpus is too easy** — six files, and all three
strategies hit 100% retrieval recall. Point it at a real documentation set
(`python -m rag ingest-all --rebuild /path/to/docs`) before drawing any
conclusion about which chunking strategy wins.

## Layout

```
src/rag/
  models.py          data structures, AnchorResolver, id generation
  config.py          every tunable, chunk-size validation
  embeddings.py      local embedder + tokenizer surface
  dedup.py           near-duplicate removal
  pipeline.py        ingest / search orchestration, index-sync assertion
  retrieval.py       RRF fusion, HybridRetriever                    [P2]
  rerank.py          cross-encoder reranking                        [P2]
  generation.py      grounded prompt, claim + citation parsing      [P3]
  verification.py    citation verification                          [P3]
  confidence.py      composite confidence scoring                   [P3]
  answering.py       AnswerPipeline — the single entry point        [P3]
  cli.py             10 commands
  loaders/           per-format loaders + dispatch
  chunking/          base, splitting, fixed, header, semantic, builder
  stores/            vector (Chroma), bm25, documents
  llm/               base seam, ollama, free_api, stub, registry    [P3]
  eval/              golden_set.yaml, dataset, metrics, runner      [P4]
  api/               FastAPI app + dashboard                        [P5]
docs/                one file per phase
scripts/             make_sample_pdf.py (regenerates the PDF fixture)
tests/               95 tests
```
