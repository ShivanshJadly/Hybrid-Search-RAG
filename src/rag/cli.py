"""Command line entry point.

    python -m rag ingest --strategy header --rebuild
    python -m rag ingest-all --rebuild
    python -m rag search "how do I roll back a deploy"
    python -m rag compare "how do I roll back a deploy"
    python -m rag status
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Quieten HF's startup chatter before the model layer is imported.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .chunking.registry import STRATEGIES  # noqa: E402
from .config import PROJECT_ROOT, settings  # noqa: E402
from .embeddings import LocalEmbedder  # noqa: E402
from .loaders.registry import SUPPORTED_EXTENSIONS, discover  # noqa: E402
from .pipeline import ingest as run_ingest  # noqa: E402
from .pipeline import search as run_search  # noqa: E402
from .stores.bm25 import Bm25Store  # noqa: E402
from .stores.documents import DocumentStore  # noqa: E402
from .stores.vector import VectorStore  # noqa: E402

app = typer.Typer(add_completion=False, help="Local hybrid-search RAG pipeline.")
console = Console()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _report_panel(report) -> Panel:
    t = Table(show_header=False, box=None, pad_edge=False)
    t.add_row("Files found", str(report.files_found))
    t.add_row("Files loaded", str(report.files_loaded))
    t.add_row("Documents stored (raw+processed)", str(report.documents_stored))
    t.add_row("Chunks built", str(report.chunks_built))
    t.add_row("Exact duplicates dropped", str(report.exact_duplicates))
    t.add_row("Near duplicates dropped", str(report.near_duplicates))
    t.add_row("Chunks indexed", str(report.chunks_indexed))
    t.add_row("Largest chunk (tokens)", str(report.max_chunk_tokens))
    t.add_row("Chroma total", str(report.vector_count))
    t.add_row("BM25 total", str(report.bm25_count))
    return Panel(t, title=f"Ingest report [{report.strategy}]", expand=False)


def _print_report(report) -> None:
    console.print(_report_panel(report))
    for name, err in report.files_failed:
        console.print(f"[yellow]skipped[/yellow] {name}: {err}")
    for src in report.empty_documents:
        console.print(f"[yellow]no extractable text[/yellow] {src}")
    for kept, dropped, sim in report.duplicate_examples:
        console.print(f"[dim]near-dup {sim:.3f}: kept {kept} / dropped {dropped}[/dim]")
    if report.in_sync:
        console.print("[green]Indexes in sync.[/green]")


def _citation(meta: dict) -> str:
    cite = meta.get("source_name", "?")
    if meta.get("heading_path"):
        cite += f" > {meta['heading_path']}"
    if meta.get("page"):
        cite += f" > p.{meta['page']}"
    return cite


@app.command()
def ingest(
    path: Optional[Path] = typer.Argument(None, help="Folder of documents."),
    strategy: str = typer.Option("fixed", "--strategy", "-s", help=str(STRATEGIES)),
    rebuild: bool = typer.Option(False, "--rebuild", help="Wipe indexes first."),
) -> None:
    """Load, chunk, embed and index a folder of documents."""
    root = (path or settings.data_dir).resolve()
    if not root.exists():
        console.print(f"[red]No such folder:[/red] {root}")
        raise typer.Exit(1)

    console.print(f"Ingesting [cyan]{root}[/cyan] with strategy [cyan]{strategy}[/cyan]")
    embedder = LocalEmbedder(settings)
    with console.status("Loading embedding model..."):
        embedder.validate()
    console.print(
        f"Model [green]{settings.embedding_model}[/green] "
        f"({embedder.dimension}-dim, max {embedder.max_seq_length} tokens)"
    )

    _print_report(run_ingest(root, settings, embedder, strategy, rebuild=rebuild))


@app.command("ingest-all")
def ingest_all(
    path: Optional[Path] = typer.Argument(None, help="Folder of documents."),
    rebuild: bool = typer.Option(False, "--rebuild"),
) -> None:
    """Index the same corpus under every chunking strategy.

    This is what Phase 4 evaluates against -- three parallel indexes over
    identical source material, differing only in where the cuts fall.
    """
    root = (path or settings.data_dir).resolve()
    if not root.exists():
        console.print(f"[red]No such folder:[/red] {root}")
        raise typer.Exit(1)

    embedder = LocalEmbedder(settings)
    with console.status("Loading embedding model..."):
        embedder.validate()

    summary = Table("strategy", "chunks", "largest (tok)", "exact dup", "near dup")
    for name in STRATEGIES:
        console.print(f"\n[bold cyan]{name}[/bold cyan]")
        report = run_ingest(root, settings, embedder, name, rebuild=rebuild)
        _print_report(report)
        summary.add_row(
            name,
            str(report.chunks_indexed),
            str(report.max_chunk_tokens),
            str(report.exact_duplicates),
            str(report.near_duplicates),
        )

    console.print("\n")
    console.print(Panel(summary, title="Strategy comparison", expand=False))


@app.command()
def search(
    query: str = typer.Argument(..., help="What to look for."),
    strategy: str = typer.Option("fixed", "--strategy", "-s"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
) -> None:
    """Query both indexes and show what each half of hybrid search returns."""
    embedder = LocalEmbedder(settings)
    results = run_search(query, settings, embedder, strategy, top_k)

    for label, key in (("Dense (meaning)", "dense"), ("Sparse (keywords)", "sparse")):
        console.print(f"\n[bold]{label}[/bold]")
        hits = results[key]
        if not hits:
            console.print("  [dim]no results[/dim]")
            continue
        for i, hit in enumerate(hits, 1):
            snippet = " ".join(hit["text"].split())[:170]
            console.print(
                f"  [cyan]{i}.[/cyan] [dim]{hit['score']:.4f}[/dim]  "
                f"{_citation(hit['meta'])}"
            )
            console.print(f"     [dim]{snippet}...[/dim]")


@app.command()
def compare(
    query: str = typer.Argument(..., help="What to look for."),
    top_k: int = typer.Option(3, "--top-k", "-k"),
) -> None:
    """Run one query against all three strategies, side by side.

    A qualitative check only. Phase 4 replaces this with scored metrics over a
    golden question set -- eyeballing results is how you form a hypothesis, not
    how you decide which strategy wins.
    """
    embedder = LocalEmbedder(settings)
    for name in STRATEGIES:
        console.print(f"\n[bold cyan]{name}[/bold cyan]")
        try:
            results = run_search(query, settings, embedder, name, top_k)
        except Exception as exc:
            console.print(f"  [dim]unavailable ({exc})[/dim]")
            continue
        hits = results["dense"]
        if not hits:
            console.print("  [dim]not indexed -- run: python -m rag ingest-all[/dim]")
            continue
        for i, hit in enumerate(hits, 1):
            snippet = " ".join(hit["text"].split())[:130]
            tokens = hit["meta"].get("n_tokens", "?")
            console.print(
                f"  [cyan]{i}.[/cyan] [dim]{hit['score']:.4f}[/dim] "
                f"[dim]({tokens} tok)[/dim] {_citation(hit['meta'])}"
            )
            console.print(f"     [dim]{snippet}...[/dim]")


@app.command()
def docs() -> None:
    """List stored documents: raw and processed sizes, anchor counts.

    A large gap between raw and processed characters is normal for HTML (markup
    is discarded) and suspicious for Markdown (little should be removed).
    """
    store = DocumentStore(settings.documents_dir)
    rows = store.list_documents()
    if not rows:
        console.print("[dim]No documents stored. Run: python -m rag ingest[/dim]")
        return

    table = Table("source", "type", "raw chars", "text chars", "anchors")
    for r in rows:
        table.add_row(
            r["source"],
            r["media_type"],
            f"{r['raw_chars']:,}",
            f"{r['text_chars']:,}",
            str(r["anchors"]),
        )
    console.print(table)


@app.command()
def status() -> None:
    """Show what is currently indexed, per strategy."""
    console.print(f"Data dir    : {settings.data_dir}")
    console.print(f"Storage dir : {settings.storage_dir}")
    console.print(f"Extensions  : {', '.join(SUPPORTED_EXTENSIONS)}")
    if settings.data_dir.exists():
        console.print(f"Files ready : {len(discover(settings.data_dir))}")
    console.print(
        f"Documents   : {DocumentStore(settings.documents_dir).count()} stored"
    )

    console.print("")
    for strategy in STRATEGIES:
        try:
            v = VectorStore(settings.chroma_dir, strategy).count()
            b = Bm25Store(settings.bm25_dir, strategy)
            b.load()
            if v == 0 and b.count() == 0:
                console.print(f"  {strategy:9s} [dim]not indexed[/dim]")
                continue
            flag = "[green]in sync[/green]" if v == b.count() else "[red]DRIFT[/red]"
            console.print(f"  {strategy:9s} chroma={v:<5d} bm25={b.count():<5d} {flag}")
        except Exception as exc:
            console.print(f"  {strategy:9s} [dim]unavailable ({exc})[/dim]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question."),
    strategy: str = typer.Option("header", "--strategy", "-s"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    backend: str = typer.Option(None, "--backend", "-b", help="ollama|groq|gemini|stub"),
    judge: str = typer.Option(None, "--judge", "-j"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="hybrid|dense|sparse"),
    no_rerank: bool = typer.Option(False, "--no-rerank"),
    no_verify: bool = typer.Option(False, "--no-verify"),
) -> None:
    """Answer a question with citations, verification and a confidence score."""
    from .answering import AnswerPipeline
    from .llm.registry import build_judge, build_llm, describe_backend

    embedder = LocalEmbedder(settings)
    llm = build_llm(settings, backend)
    judge_client = build_judge(settings, judge)

    if not llm.health():
        console.print(
            f"[yellow]Generator {describe_backend(llm)} is not reachable.[/yellow] "
            "Falling back to the extractive stub - answers will be crude but the "
            "pipeline is exercised end to end. See README 'Turning it on'."
        )
        llm = build_llm(settings, "stub", mode="extractive")

    pipeline = AnswerPipeline(settings, embedder, llm, judge_client, strategy=strategy)
    answer = pipeline.ask(
        question,
        top_n=top_k,
        use_dense=mode != "sparse",
        use_sparse=mode != "dense",
        rerank=not no_rerank,
        verify_citations=not no_verify,
    )

    console.print()
    console.print(Panel(answer.text, title="Answer", expand=False,
                        border_style="yellow" if answer.declined else "green"))

    c = answer.confidence
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"
    heading = "Refusal confidence" if c.kind == "refusal" else "Confidence"
    console.print(
        f"\n[bold]{heading}[/bold] {c.composite:.0%} ({c.label})  "
        f"[dim]retrieval {pct(c.retrieval)} | coverage {pct(c.coverage)} | "
        f"citations {pct(c.citation_accuracy)}[/dim]"
    )
    for reason in c.reasons:
        console.print(f"  [dim]- {reason}[/dim]")

    console.print("\n[bold]Sources[/bold]")
    for i, s in enumerate(answer.sources, 1):
        score = f"{s.rerank_score:+.3f}" if s.rerank_score is not None else f"{s.rrf_score:.5f}"
        console.print(f"  [cyan][{i}][/cyan] [dim]{score} {s.found_by:6s}[/dim] {s.citation()}")

    if answer.verification.checks:
        console.print("\n[bold]Citation checks[/bold]")
        for chk in answer.verification.checks:
            colour = "green" if chk.verdict.is_ok else "yellow"
            console.print(
                f"  [{colour}]{chk.verdict.value:11s}[/{colour}] "
                f"[dim]{chk.claim[:70]}...[/dim]"
            )


@app.command("eval")
def run_eval(
    strategy: str = typer.Option(None, "--strategy", "-s", help="default: all three"),
    limit: int = typer.Option(None, "--limit", "-n", help="first N questions only"),
    backend: str = typer.Option(None, "--backend", "-b"),
    judge: str = typer.Option(None, "--judge", "-j"),
    no_grade: bool = typer.Option(False, "--no-grade", help="skip judge-backed metrics"),
) -> None:
    """Run the golden question set and report metrics."""
    from pathlib import Path as _Path

    from .eval.dataset import load_golden_set, summarise
    from .eval.runner import sweep, write_report
    from .llm.registry import build_judge, build_llm

    golden = load_golden_set()
    console.print(f"Golden set: {summarise(golden)}")

    embedder = LocalEmbedder(settings)
    llm = build_llm(settings, backend)
    judge_client = build_judge(settings, judge)
    if not llm.health():
        console.print("[yellow]Generator unreachable; using extractive stub.[/yellow]")
        llm = build_llm(settings, "stub", mode="extractive")

    strategies = [strategy] if strategy else list(STRATEGIES)
    total = len(golden) if not limit else min(limit, len(golden))

    with console.status("Evaluating...") as status:
        def progress(i, n, q):
            status.update(f"[{i}/{n}] {q.id} {q.question[:50]}")
        summaries = sweep(
            settings, embedder, llm, judge_client, strategies,
            limit=limit, grade=not no_grade, on_progress=progress,
        )

    table = Table("strategy", "recall", "precision", "correct", "faithful",
                  "citations", "refusal", "false refusal")
    for s in summaries:
        d = s.as_dict()
        table.add_row(
            d["strategy"],
            f"{d['retrieval_recall']:.1%}",
            f"{d['retrieval_precision']:.1%}",
            f"{d['correctness']:.1%}",
            f"{d['faithfulness']:.1%}",
            f"{d['citation_accuracy']:.1%}",
            f"{d['refusal_accuracy']:.1%}",
            f"{d['false_refusal_rate']:.1%}",
        )
    console.print(Panel(table, title=f"Evaluation ({total} questions)", expand=False))

    path = write_report(summaries, _Path(settings.storage_dir) / "eval")
    console.print(f"[dim]Report written to {path}[/dim]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the API and dashboard."""
    import uvicorn

    console.print(f"Dashboard: [cyan]http://{host}:{port}/[/cyan]")
    console.print(f"API docs:  [cyan]http://{host}:{port}/docs[/cyan]")
    uvicorn.run("rag.api.app:app", host=host, port=port, reload=reload)


@app.command()
def check() -> None:
    """Verify backends are configured and reachable, before a long run.

    Checks in the order things actually fail: is a key present, does it work,
    is the configured model reachable with it, and does a real call return
    text. Each step names the fix rather than just reporting failure.
    """
    from .llm.free_api import GeminiClient, GroqClient
    from .llm.registry import build_judge, build_llm, describe_backend

    console.print(f"[bold]Config[/bold]  generator=[cyan]{settings.llm_backend}[/cyan]  "
                  f"judge=[cyan]{settings.judge_backend}[/cyan]")
    console.print(f"[dim].env expected at: {PROJECT_ROOT / '.env'}[/dim]\n")

    ok = True
    for role, client in (("generator", build_llm(settings)), ("judge", build_judge(settings))):
        console.print(f"[bold]{role}[/bold] - {describe_backend(client)}")

        if isinstance(client, (GeminiClient, GroqClient)):
            if not client.api_key:
                provider = "GEMINI" if isinstance(client, GeminiClient) else "GROQ"
                console.print(
                    f"  [red]no API key[/red] - add "
                    f"[cyan]{provider}_API_KEY=...[/cyan] to .env"
                )
                ok = False
                console.print()
                continue
            console.print(f"  [green]key present[/green] ({client.api_key[:6]}...)")

        if isinstance(client, GeminiClient):
            try:
                names = client.list_models()
            except Exception as exc:
                console.print(f"  [red]key rejected by Google[/red]: {exc}")
                console.print("  [dim]Get a key at aistudio.google.com/apikey[/dim]")
                ok = False
                console.print()
                continue

            console.print(f"  [green]key valid[/green] - {len(names)} models reachable")
            if client.model in names:
                console.print(f"  [green]model {client.model} available[/green]")
            else:
                ok = False
                flash = [n for n in names if "flash" in n and "preview" not in n][:6]
                console.print(f"  [red]model {client.model} NOT available to this key[/red]")
                console.print(f"  [dim]try one of: {', '.join(flash) or ', '.join(names[:6])}[/dim]")
                console.print("  [dim]set RAG_GEMINI_MODEL=<id> in .env[/dim]")

        try:
            reply = client.complete("Reply with the single word OK.", "Say OK.", max_tokens=16)
            if reply.text.strip():
                console.print(f"  [green]live call ok[/green] - replied {reply.text.strip()[:40]!r}")
            else:
                console.print("  [yellow]live call returned empty text[/yellow] "
                              "(safety filter, or the model produced nothing)")
                ok = False
        except Exception as exc:
            console.print(f"  [red]live call failed[/red]: {str(exc)[:160]}")
            ok = False
        console.print()

    if ok:
        console.print("[green]All backends ready.[/green] Try: "
                      "[cyan]python -m rag ask \"how do I roll back a deploy\"[/cyan]")
    else:
        console.print("[yellow]Fix the items above, then re-run "
                      "[cyan]python -m rag check[/cyan].[/yellow]")
        raise typer.Exit(1)
