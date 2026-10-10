"""FastAPI service.

Endpoints follow the plan: POST /v1/ask, GET /v1/documents, POST /v1/ingest.
The dashboard is served from the same app at /, so there is no separate
frontend service to run or deploy.

Heavy objects -- the embedding model, the reranker, the retriever -- are built
once at startup and reused. Building them per request would load ~200MB of
model weights on every question.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..answering import AnswerPipeline
from ..chunking.registry import STRATEGIES
from ..config import settings
from ..embeddings import LocalEmbedder
from ..llm.registry import build_judge, build_llm, describe_backend
from ..pipeline import ingest as run_ingest
from ..rerank import build_reranker
from ..stores.bm25 import Bm25Store
from ..stores.documents import DocumentStore
from ..stores.vector import VectorStore

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"

# Populated at startup; one pipeline per chunking strategy.
_state: dict = {"pipelines": {}, "embedder": None, "llm": None, "judge": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = LocalEmbedder(settings)
    embedder.validate()
    reranker = build_reranker(settings)
    llm = build_llm(settings)
    judge = build_judge(settings)

    _state["embedder"] = embedder
    _state["llm"] = llm
    _state["judge"] = judge
    for strategy in STRATEGIES:
        _state["pipelines"][strategy] = AnswerPipeline(
            settings, embedder, llm, judge, strategy=strategy, reranker=reranker
        )
    log.info("Ready. generator=%s judge=%s", describe_backend(llm), describe_backend(judge))
    yield
    _state["pipelines"].clear()


app = FastAPI(
    title="Hybrid RAG over Internal Docs",
    version="0.5.0",
    description="Hybrid dense+sparse retrieval with verified inline citations.",
    lifespan=lifespan,
)


# ----------------------------------------------------------------- schemas


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    strategy: str = Field("header", description=f"one of {STRATEGIES}")
    top_k: int = Field(5, ge=1, le=20)
    use_dense: bool = True
    use_sparse: bool = True
    rerank: Optional[bool] = None
    verify_citations: bool = True


class IngestRequest(BaseModel):
    path: Optional[str] = None
    strategy: str = "header"
    rebuild: bool = False


# ---------------------------------------------------------------- routes


@app.get("/healthz")
def healthz() -> dict:
    """Liveness plus which backends are actually reachable.

    The LLM check matters: without a generator the service still retrieves,
    but every answer degrades to a decline, and that should be visible here
    rather than discovered one question at a time.
    """
    llm = _state.get("llm")
    judge = _state.get("judge")
    return {
        "status": "ok",
        "generator": describe_backend(llm) if llm else None,
        "generator_reachable": bool(llm and llm.health()),
        "judge": describe_backend(judge) if judge else None,
        "judge_reachable": bool(judge and judge.health()),
        "strategies": list(STRATEGIES),
    }


@app.post("/v1/ask")
def ask(req: AskRequest) -> dict:
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"unknown strategy '{req.strategy}'")
    if not (req.use_dense or req.use_sparse):
        raise HTTPException(400, "at least one of use_dense/use_sparse must be true")

    pipeline = _state["pipelines"].get(req.strategy)
    if pipeline is None:
        raise HTTPException(503, "service still starting")

    try:
        answer = pipeline.ask(
            req.question,
            top_n=req.top_k,
            use_dense=req.use_dense,
            use_sparse=req.use_sparse,
            rerank=req.rerank,
            verify_citations=req.verify_citations,
        )
    except Exception as exc:
        log.exception("ask failed")
        raise HTTPException(500, f"answering failed: {exc}") from exc

    return answer.as_dict()


@app.get("/v1/documents")
def documents() -> dict:
    store = DocumentStore(settings.documents_dir)
    rows = store.list_documents()
    index = {}
    for strategy in STRATEGIES:
        try:
            vector = VectorStore(settings.chroma_dir, strategy).count()
            bm25 = Bm25Store(settings.bm25_dir, strategy)
            bm25.load()
            index[strategy] = {
                "chroma": vector,
                "bm25": bm25.count(),
                "in_sync": vector == bm25.count(),
            }
        except Exception as exc:
            index[strategy] = {"error": str(exc)}
    return {"documents": rows, "indexes": index}


@app.post("/v1/ingest")
def ingest(req: IngestRequest, background: BackgroundTasks) -> dict:
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"unknown strategy '{req.strategy}'")
    root = Path(req.path).resolve() if req.path else settings.data_dir
    if not root.exists():
        raise HTTPException(400, f"path does not exist: {root}")

    embedder = _state.get("embedder")
    if embedder is None:
        raise HTTPException(503, "service still starting")

    # Ingestion is slow enough to exceed a request timeout on a real corpus,
    # so it runs in the background and the caller polls /v1/documents.
    background.add_task(
        run_ingest, root, settings, embedder, req.strategy, req.rebuild
    )
    return {
        "status": "started",
        "path": str(root),
        "strategy": req.strategy,
        "poll": "/v1/documents",
    }


# ------------------------------------------------------------- dashboard

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def dashboard():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return {"message": "dashboard not built", "docs": "/docs"}
    return FileResponse(str(index))
