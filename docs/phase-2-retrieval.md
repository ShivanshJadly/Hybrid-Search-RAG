# Phase 2 — Hybrid Retrieval Engine

**Plan reference:** Notion, Phase 2 (Day 3–6). **Status: complete.**

Turns two separate ranked lists into one, then lets a cross-encoder pick the
final five.

```
question
  → dense retrieval   (bge embedding → ChromaDB cosine, top 10)
  → sparse retrieval  (BM25, top 10)
  → RRF fusion        (weighted 0.7 / 0.3, k=60)
  → top 20 candidates
  → cross-encoder rerank
  → top 5
```

## Files

| File | Role |
|---|---|
| `retrieval.py` | `reciprocal_rank_fusion()`, `HybridRetriever`, `RetrievedChunk`, `RetrievalTrace` |
| `rerank.py` | `CrossEncoderReranker`, `NoOpReranker`, `build_reranker()` |

## Why fuse on rank, not score

The two retrievers produce numbers that mean different things. Cosine
similarity is bounded roughly 0–1. BM25 is unbounded and scales with corpus
statistics — on this corpus a BM25 score of 9.26 and 2.07 appeared for the same
query. Normalising them onto a shared scale needs assumptions that break
whenever the corpus changes.

Ranks are comparable by construction. That is the entire appeal of RRF:

```
score(d) = Σ over lists L:  weight(L) / (k + rank(d in L))
```

A document missing from a list contributes nothing from it. `k` (default 60,
from the original paper) damps the advantage of the very top ranks.

Pinned by `test_rrf_uses_rank_not_score`: multiplying a BM25 score by 5000 does
not change the fused ordering.

## Provenance is recorded, not inferred

Every `RetrievedChunk` carries `dense_rank`, `sparse_rank`, and a derived
`found_by` of `dense` / `sparse` / `both`. This is what makes "hybrid beat
dense-only" a measurable claim rather than an assertion, and it drives the
dashboard's per-chunk tags.

On the sample corpus, a typical query has **7–8 of 12 candidates found by both
retrievers** — the overlap is high because the corpus is small and topically
narrow. Expect it to fall on a real corpus, which is where fusion starts
earning its place.

## Reranking

The retrievers score query and chunk *separately* and compare vectors. A
cross-encoder reads both together in one pass, so it judges actual relevance.
Far more accurate, far too slow for a whole corpus — hence retrieve 20 cheaply,
rerank those 20 properly, keep 5.

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80MB, local, free.

The plan offers "cross-encoder / LLM-as-judge". The cross-encoder wins here
because it costs nothing per query, and the judge budget is better spent on
citation verification where nothing cheaper exists.

**Scores are raw logits**, roughly −11 to +11. They order results correctly and
mean nothing as absolute numbers. `confidence.py` squashes them through a
logistic when a bounded value is actually needed. A reranked result showing
`-5.35` is not "negative confidence" — it is rank 2 of 5.

## Ablation support

`retrieve()` takes `use_dense` / `use_sparse` / `rerank` flags. The dashboard
toggle, the CLI `--mode`, and any evaluation ablation all go through this one
path — an ablation that runs different code is not measuring the same system.

## Try it

```bash
python -m rag search "CACHE_TTL_JITTER_PERCENT"     # both lists, unfused
python -m rag ask "how do I roll back" --mode dense # dense only
python -m rag compare "how do I roll back"          # across strategies
```

## Limitations

- **Fusion weights (0.7/0.3) are untuned.** Phase 4 has the machinery to sweep
  them; that sweep has not been run.
- **`rerank_candidates=20` is a guess**, as is `top_n=5`. Both are config
  values, neither is measured.
- **The reranker loads ~80MB on first use** and runs on CPU. Roughly 20 pairs
  per query is fine; a much larger candidate set would need batching work.
- **No query rewriting.** A question phrased very differently from the document
  wording relies entirely on the embedding model bridging the gap.
