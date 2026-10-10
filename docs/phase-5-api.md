# Phase 5 — API & Dashboard

**Plan reference:** Notion, Phase 5 (Day 11–13). **Status: complete; Docker
image not built** (see Limitations).

## Files

| File | Role |
|---|---|
| `api/app.py` | FastAPI service, lifespan-managed model loading |
| `api/static/index.html` | Dashboard — single file, no build step |
| `Dockerfile` | CPU-only image with models baked in |
| `docker-compose.yml` | One service + persistent volumes |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/ask` | Answer + citations + confidence + sources + retrieval trace |
| `GET` | `/v1/documents` | Stored documents and per-strategy index counts |
| `POST` | `/v1/ingest` | Kick off ingestion in the background |
| `GET` | `/healthz` | Liveness **and** whether the generator/judge are reachable |
| `GET` | `/` | Dashboard |
| `GET` | `/docs` | OpenAPI |

`/v1/ask` accepts `strategy`, `top_k`, `use_dense`, `use_sparse`, `rerank` and
`verify_citations` — the ablation flags, so the dashboard toggle and any
experiment share the production code path.

`/healthz` reports backend reachability because without a generator the service
still retrieves but every answer degrades to a decline. That should be visible
at a glance, not discovered one question at a time.

Ingestion runs in a `BackgroundTask` — a real corpus takes longer than a
request timeout — and the caller polls `/v1/documents`.

```bash
curl -X POST localhost:8000/v1/ask -H 'Content-Type: application/json' \
  -d '{"question":"how long are audit trails kept","strategy":"header"}'
```

## Dashboard

`python -m rag serve` → <http://127.0.0.1:8000/>

Everything the plan asks for:

- **Answer with clickable citations** — `[1]` renders as a chip; clicking
  scrolls to that chunk, highlights it and expands it.
- **Ranked retrieved chunks** with `both` / `dense` / `sparse` tags, rerank or
  RRF score, and the rank each retriever gave it.
- **Confidence breakdown** — four meters plus the plain-English reasons.
- **Hybrid vs dense toggle**, plus sparse-only, strategy selector, rerank and
  verification switches, and top-k.

Single HTML file, vanilla JS, no build step, no npm. Theme-aware, and no
external requests. Verified rendering in-browser.

## Running

```bash
python -m rag serve                       # local
python -m rag ingest-all --rebuild        # rebuild all indexes first
docker compose up --build                 # containerised
```

Backends are environment-driven (`.env.example` has the full list):

```bash
RAG_LLM_BACKEND=ollama RAG_JUDGE_BACKEND=groq python -m rag serve
```

## One deliberate deviation from the plan

The plan sketches **"docker-compose (API + ChromaDB + frontend)"**. This build
ships **one service**, because:

- ChromaDB runs **embedded** — a file-backed library, not a server. A separate
  container would mean switching `PersistentClient` to `HttpClient` and adding
  a network hop for no capability gain.
- The dashboard is **static files served by the API**, so there is no frontend
  process to run.

Three containers here would be ceremony, not architecture. If Chroma is ever
needed as a shared server, swap the client in `stores/vector.py` and add the
`chromadb/chroma` service — the compose file says so in a comment.

## Limitations

- **The Docker image has never been built or run.** The Dockerfile and compose
  file are written and internally consistent, but unverified. Expect to debug
  the first `docker compose up`.
- **No auth, no rate limiting, no CORS policy.** Fine for localhost, not for
  anything exposed.
- **`/v1/ingest` returns immediately with no job id.** Progress is inferred by
  polling `/v1/documents`; a failure mid-ingest is only visible in server logs.
- **Model loading happens at startup**, so the first boot after a fresh install
  downloads ~200MB. The Dockerfile bakes them in to avoid this in containers.
- **Single worker assumed.** Multiple uvicorn workers would each load their own
  copy of the models.
