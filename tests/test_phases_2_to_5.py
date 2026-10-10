"""Tests for retrieval, generation, evaluation and the API.

No model weights are loaded and no network is touched: fusion is pure maths,
generation runs against the stub, and the API is exercised through FastAPI's
TestClient with the pipeline replaced.
"""

from __future__ import annotations

import pytest
import yaml

from rag.answering import Answer, AnswerPipeline
from rag.confidence import (
    CITATION_WEIGHT,
    COVERAGE_WEIGHT,
    RETRIEVAL_WEIGHT,
    citation_coverage,
    retrieval_confidence,
)
from rag.confidence import score as score_confidence
from rag.config import Settings
from rag.eval.dataset import InvalidGoldenSet, load_golden_set, summarise
from rag.eval.metrics import EvalSummary, QuestionResult, score_retrieval
from rag.generation import (
    INSUFFICIENT,
    GeneratedAnswer,
    build_context_block,
    build_prompt,
    generate,
    parse_claims,
)
from rag.llm.registry import UnknownBackend, build_llm
from rag.llm.stub import StubLLM
from rag.rerank import NoOpReranker
from rag.retrieval import RetrievedChunk, reciprocal_rank_fusion
from rag.verification import Verdict, VerificationReport, verify


def hit(cid: str, text: str = "some text", score: float = 0.5, **meta) -> dict:
    base = {"source_name": f"{cid}.md", "heading_path": "", "page": 0}
    base.update(meta)
    return {"chunk_id": cid, "text": text, "meta": base, "score": score}


def chunk(cid: str = "c1", text: str = "text", **kw) -> RetrievedChunk:
    meta = {"source_name": kw.pop("source_name", f"{cid}.md"), "heading_path": "", "page": 0}
    return RetrievedChunk(chunk_id=cid, text=text, meta=meta, **kw)


# ============================================================ PHASE 2


def test_rrf_ranks_a_chunk_found_by_both_above_singletons():
    """The core claim of hybrid search: agreement between two different
    retrievers is evidence, and fusion should reward it."""
    dense = [hit("a"), hit("b")]
    sparse = [hit("c"), hit("a")]
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    assert fused[0].chunk_id == "a"
    assert fused[0].found_by == "both"


def test_rrf_uses_rank_not_score():
    """Scores from the two retrievers are on incomparable scales; feeding a
    wildly larger BM25 score must not change the fusion outcome."""
    a = reciprocal_rank_fusion([hit("x", score=0.9)], [hit("y", score=2.0)], k=60)
    b = reciprocal_rank_fusion([hit("x", score=0.9)], [hit("y", score=9999.0)], k=60)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_rrf_weights_shift_the_winner():
    dense_heavy = reciprocal_rank_fusion(
        [hit("d")], [hit("s")], dense_weight=0.9, sparse_weight=0.1
    )
    sparse_heavy = reciprocal_rank_fusion(
        [hit("d")], [hit("s")], dense_weight=0.1, sparse_weight=0.9
    )
    assert dense_heavy[0].chunk_id == "d"
    assert sparse_heavy[0].chunk_id == "s"


def test_rrf_records_provenance_from_each_list():
    fused = reciprocal_rank_fusion([hit("a"), hit("b")], [hit("b")], k=60)
    by_id = {c.chunk_id: c for c in fused}
    assert by_id["a"].found_by == "dense"
    assert by_id["a"].sparse_rank is None
    assert by_id["b"].found_by == "both"
    assert by_id["b"].dense_rank == 2 and by_id["b"].sparse_rank == 1


def test_rrf_handles_one_empty_list():
    assert len(reciprocal_rank_fusion([hit("a")], [], k=60)) == 1
    assert reciprocal_rank_fusion([], [], k=60) == []


def test_larger_k_flattens_the_ranking():
    """k damps the advantage of top ranks -- that is its whole job."""
    small = reciprocal_rank_fusion([hit("a"), hit("b")], [], k=1)
    large = reciprocal_rank_fusion([hit("a"), hit("b")], [], k=1000)
    gap_small = small[0].rrf_score - small[1].rrf_score
    gap_large = large[0].rrf_score - large[1].rrf_score
    assert gap_small > gap_large


def test_final_score_prefers_the_reranker():
    c = chunk()
    c.rrf_score = 0.01
    assert c.final_score == 0.01
    c.rerank_score = 4.2
    assert c.final_score == 4.2


def test_noop_reranker_preserves_order():
    chunks = [chunk("a"), chunk("b")]
    assert NoOpReranker().rerank("q", chunks) == chunks


# ============================================================ PHASE 3


def test_context_block_is_numbered_from_one():
    block = build_context_block([chunk("a", "first"), chunk("b", "second")])
    assert block.startswith("[1]")
    assert "[2]" in block


def test_parse_claims_extracts_citations_per_sentence():
    claims = parse_claims("Rollbacks reuse the deploy path [1]. They skip canary [2][3].")
    assert len(claims) == 2
    assert claims[0].citation_ids == [1]
    assert claims[1].citation_ids == [2, 3]


def test_parse_claims_flags_uncited_sentences():
    claims = parse_claims("This is cited [1]. This one is not.")
    assert claims[0].is_cited
    assert not claims[1].is_cited


def test_dangling_citation_is_detected():
    answer = GeneratedAnswer(
        question="q",
        answer="Claim [5].",
        claims=parse_claims("Claim [5]."),
        sources=[chunk("a")],
    )
    assert answer.dangling_citations == [5]
    assert answer.source_for(5) is None
    assert answer.source_for(1) is not None


def test_generate_declines_when_nothing_retrieved():
    answer = generate("q", [], StubLLM())
    assert answer.insufficient
    assert INSUFFICIENT in answer.answer


def test_generate_respects_an_insufficient_reply():
    llm = StubLLM(mode="scripted", responses=[f"{INSUFFICIENT} No pricing data."])
    answer = generate("q", [chunk()], llm)
    assert answer.insufficient
    assert answer.claims == []


def test_verify_marks_dangling_citations_without_calling_the_judge():
    judge = StubLLM(mode="judge")
    answer = GeneratedAnswer(
        question="q",
        answer="Claim [9].",
        claims=parse_claims("Claim about something [9]."),
        sources=[chunk("a")],
    )
    report = verify(answer, judge)
    assert [c.verdict for c in report.checks] == [Verdict.DANGLING]
    assert judge.calls == [], "a dangling citation needs no judge call"


def test_verify_flags_uncited_factual_sentences():
    judge = StubLLM(mode="judge")
    answer = GeneratedAnswer(
        question="q",
        answer="The pool size defaults to twenty per instance.",
        claims=parse_claims("The pool size defaults to twenty per instance."),
        sources=[chunk("a")],
    )
    report = verify(answer, judge)
    assert report.checks[0].verdict is Verdict.UNCITED


def test_verify_ignores_pleasantries():
    """Without this filter, 'I hope that helps.' counts against coverage."""
    judge = StubLLM(mode="judge")
    answer = GeneratedAnswer(
        question="q",
        answer="Supported claim here [1]. I hope that helps.",
        claims=parse_claims("Supported claim here [1]. I hope that helps."),
        sources=[chunk("a")],
    )
    report = verify(answer, judge)
    assert all(c.verdict is not Verdict.UNCITED for c in report.checks)


def test_unparseable_judge_reply_does_not_count_as_supported():
    """Failing open here would silently inflate citation accuracy."""
    judge = StubLLM(mode="scripted", responses=["I'm not sure about this one."])
    answer = GeneratedAnswer(
        question="q",
        answer="A claim [1].",
        claims=parse_claims("A claim about the system [1]."),
        sources=[chunk("a")],
    )
    report = verify(answer, judge)
    assert report.checks[0].verdict is Verdict.PARTIAL
    assert report.accuracy < 1.0


def test_verification_accuracy_with_nothing_to_check_is_one():
    assert VerificationReport(checks=[]).accuracy == 1.0


# --- confidence ---


def test_retrieval_confidence_is_zero_without_chunks():
    assert retrieval_confidence([]) == 0.0


def test_retrieval_confidence_rises_with_rerank_score():
    low, high = chunk(), chunk()
    low.rerank_score, high.rerank_score = -6.0, 6.0
    assert retrieval_confidence([low]) < 0.1
    assert retrieval_confidence([high]) > 0.9


def test_coverage_is_the_share_of_cited_sentences():
    answer = GeneratedAnswer(
        question="q",
        answer="a",
        claims=parse_claims("Cited one [1]. Uncited two. Cited three [1]."),
        sources=[chunk()],
    )
    assert citation_coverage(answer) == pytest.approx(2 / 3)


def test_fully_verified_answer_scores_higher_than_a_dangling_one():
    src = chunk()
    src.rerank_score = 5.0

    good = GeneratedAnswer("q", "Claim [1].", parse_claims("A solid claim here [1]."), [src])
    bad = GeneratedAnswer("q", "Claim [7].", parse_claims("A solid claim here [7]."), [src])

    good_score = score_confidence(good, [src], verify(good, StubLLM(mode="judge")))
    bad_score = score_confidence(bad, [src], verify(bad, StubLLM(mode="judge")))
    assert good_score.composite > bad_score.composite
    assert any("non-existent" in r for r in bad_score.reasons)


def test_confidence_reasons_are_never_empty():
    src = chunk()
    src.rerank_score = 5.0
    answer = GeneratedAnswer("q", "Claim [1].", parse_claims("A solid claim here [1]."), [src])
    result = score_confidence(answer, [src], verify(answer, StubLLM(mode="judge")))
    assert result.reasons


# --- pipeline decline behaviour ---


class _FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query, **kw):
        from rag.retrieval import RetrievalTrace

        return list(self._chunks), RetrievalTrace(query=query, strategy="fake")


def _pipeline_with(chunks, llm, judge, settings=None):
    p = AnswerPipeline.__new__(AnswerPipeline)
    p.settings = settings or Settings()
    p.llm = llm
    p.judge = judge
    p.retriever = _FakeRetriever(chunks)
    return p


def test_pipeline_declines_when_confidence_is_below_threshold():
    weak = chunk()
    weak.rerank_score = -9.0
    llm = StubLLM(mode="scripted", responses=["Something uncited and vague."])
    answer = _pipeline_with([weak], llm, StubLLM(mode="judge")).ask("q")
    assert answer.declined
    assert "could not find enough" in answer.text


def test_pipeline_answers_when_confident():
    strong = chunk(text="Audit trails are retained for seven years.")
    strong.rerank_score = 8.0
    llm = StubLLM(mode="scripted", responses=["Audit trails are kept for seven years [1]."])
    answer = _pipeline_with([strong], llm, StubLLM(mode="judge")).ask("q")
    assert not answer.declined
    assert "[1]" in answer.text


def test_answer_serialises_for_the_api():
    strong = chunk()
    strong.rerank_score = 8.0
    llm = StubLLM(mode="scripted", responses=["A claim [1]."])
    payload = _pipeline_with([strong], llm, StubLLM(mode="judge")).ask("q").as_dict()
    for key in ("question", "answer", "declined", "confidence", "sources",
                "citations", "retrieval"):
        assert key in payload
    assert payload["sources"][0]["n"] == 1


def test_unknown_backend_is_rejected():
    with pytest.raises(UnknownBackend, match="ollama"):
        build_llm(Settings(), "not-a-backend")


# ============================================================ PHASE 4


def test_golden_set_loads_and_covers_every_category():
    questions = load_golden_set()
    stats = summarise(questions)
    assert stats["total"] >= 50, "the plan asks for 50+ questions"
    for category in ("lookup", "multihop", "no_answer", "ambiguous"):
        assert stats["by_category"].get(category, 0) > 0
    assert stats["unanswerable"] >= 3


def test_golden_ids_are_unique():
    ids = [q.id for q in load_golden_set()]
    assert len(ids) == len(set(ids))


def _write_golden(tmp_path, rows):
    path = tmp_path / "g.yaml"
    path.write_text(yaml.safe_dump(rows), encoding="utf-8")
    return path


def test_answerable_question_without_sources_is_rejected(tmp_path):
    """Silently unscoreable rows would quietly deflate retrieval recall."""
    path = _write_golden(tmp_path, [
        {"id": "a", "category": "lookup", "question": "q", "answerable": True,
         "expected_sources": []}
    ])
    with pytest.raises(InvalidGoldenSet, match="expected_sources"):
        load_golden_set(path)


def test_no_answer_question_with_sources_is_rejected(tmp_path):
    path = _write_golden(tmp_path, [
        {"id": "a", "category": "no_answer", "question": "q", "answerable": False,
         "expected_sources": ["x.md"]}
    ])
    with pytest.raises(InvalidGoldenSet, match="must not list"):
        load_golden_set(path)


def test_duplicate_ids_are_rejected(tmp_path):
    row = {"id": "dup", "category": "lookup", "question": "q", "answerable": True,
           "expected_sources": ["x.md"]}
    with pytest.raises(InvalidGoldenSet, match="duplicate"):
        load_golden_set(_write_golden(tmp_path, [row, dict(row)]))


def test_unknown_category_is_rejected(tmp_path):
    path = _write_golden(tmp_path, [
        {"id": "a", "category": "nonsense", "question": "q", "answerable": True,
         "expected_sources": ["x.md"]}
    ])
    with pytest.raises(InvalidGoldenSet, match="category"):
        load_golden_set(path)


def test_retrieval_scoring_matches_on_document_name():
    """Matching by document, not chunk id, is what lets one golden set score
    all three chunking strategies."""
    from rag.eval.dataset import GoldenQuestion

    answer = Answer.__new__(Answer)
    answer.sources = [
        chunk("a", source_name="deploy-guide.md"),
        chunk("b", source_name="other.md"),
    ]
    golden = GoldenQuestion("q1", "lookup", "q", True, "a", ["deploy-guide.md"])
    hit_, precision = score_retrieval(answer, golden)
    assert hit_ is True
    assert precision == 0.5


def test_false_refusal_rate_penalises_always_declining():
    """Guards the metric set against a system that games refusal accuracy by
    declining everything."""
    summary = EvalSummary(
        strategy="s",
        results=[
            QuestionResult("1", "lookup", "q", answerable=True, declined=True),
            QuestionResult("2", "lookup", "q", answerable=True, declined=False),
            QuestionResult("3", "no_answer", "q", answerable=False,
                           declined=True, refusal_correct=True),
        ],
    )
    assert summary.refusal_accuracy == 1.0
    assert summary.false_refusal_rate == pytest.approx(0.5)


# ============================================================ PHASE 5


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from rag.api import app as api_module

    strong = chunk(text="Audit trails are retained for seven years.")
    strong.rerank_score = 7.0
    pipeline = _pipeline_with(
        [strong],
        StubLLM(mode="scripted", responses=["Seven years [1]." for _ in range(50)]),
        StubLLM(mode="judge"),
    )
    state = {
        "pipelines": {s: pipeline for s in ("fixed", "header", "semantic")},
        "embedder": object(),
        "llm": StubLLM(),
        "judge": StubLLM(mode="judge"),
    }
    monkeypatch.setattr(api_module, "_state", state)

    # Deliberately NOT used as a context manager: entering one runs the app's
    # lifespan, which would load real model weights and overwrite this fake
    # state with an Ollama-backed pipeline.
    yield TestClient(api_module.app, raise_server_exceptions=False)


def test_healthz_reports_backends(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert "strategies" in body


def test_ask_returns_the_full_payload(client):
    res = client.post("/v1/ask", json={"question": "how long are audit trails kept"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"]
    assert body["sources"][0]["citation"]
    assert "composite" in body["confidence"]


def test_ask_rejects_an_unknown_strategy(client):
    res = client.post("/v1/ask", json={"question": "hello there", "strategy": "nope"})
    assert res.status_code == 400


def test_ask_rejects_disabling_both_retrievers(client):
    res = client.post(
        "/v1/ask",
        json={"question": "hello there", "use_dense": False, "use_sparse": False},
    )
    assert res.status_code == 400


def test_ask_validates_question_length(client):
    assert client.post("/v1/ask", json={"question": "hi"}).status_code == 422


def test_documents_endpoint_reports_index_sync(client):
    body = client.get("/v1/documents").json()
    assert "documents" in body and "indexes" in body
    for strategy in ("fixed", "header", "semantic"):
        assert strategy in body["indexes"]


def test_ingest_rejects_a_missing_path(client):
    res = client.post("/v1/ingest", json={"path": "/definitely/not/here"})
    assert res.status_code == 400


def test_dashboard_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Hybrid RAG" in res.text


# --- refusals are scored on their own scale --------------------------------
#
# The blended formula only describes an answer. A refusal makes no claims and
# cites nothing, so coverage and accuracy have nothing to measure; pinning
# them at 1.0 to keep the blend whole inverted it, and a refusal issued over
# strong retrieval outscored a correct one.


def _refusal(src):
    """A generated answer that declined for want of context."""
    return GeneratedAnswer(
        question="q",
        answer="INSUFFICIENT_CONTEXT The documents do not cover this.",
        claims=[],
        sources=[src],
        insufficient=True,
    )


def test_refusal_reports_no_coverage_or_accuracy_rather_than_a_perfect_score():
    src = chunk()
    src.rerank_score = -10.0
    answer = _refusal(src)

    breakdown = score_confidence(answer, [src], verify(answer, StubLLM(mode="judge")))

    assert breakdown.kind == "refusal"
    assert breakdown.coverage is None
    assert breakdown.citation_accuracy is None
    assert breakdown.as_dict()["coverage"] is None
    assert breakdown.as_dict()["citation_accuracy"] is None


def test_a_correct_refusal_outscores_a_false_one():
    """The inversion this scale exists to fix.

    Both decline. One found nothing, the other declined while holding strong
    material -- that second one is the failure, and must score lower.
    """
    nothing, plenty = chunk(), chunk()
    nothing.rerank_score, plenty.rerank_score = -10.0, 8.0

    correct = score_confidence(
        _refusal(nothing), [nothing], verify(_refusal(nothing), StubLLM(mode="judge"))
    )
    false = score_confidence(
        _refusal(plenty), [plenty], verify(_refusal(plenty), StubLLM(mode="judge"))
    )

    assert correct.composite > false.composite
    assert correct.composite > 0.9
    assert false.composite < 0.1
    assert any("false refusal" in r for r in false.reasons)


def test_answer_scale_is_unchanged_when_every_component_is_measured():
    """Renormalising must not move the number for an ordinary answer."""
    src = chunk()
    src.rerank_score = 5.0
    good = GeneratedAnswer("q", "Claim [1].", parse_claims("A solid claim here [1]."), [src])

    breakdown = score_confidence(good, [src], verify(good, StubLLM(mode="judge")))
    expected = (
        RETRIEVAL_WEIGHT * breakdown.retrieval
        + COVERAGE_WEIGHT * breakdown.coverage
        + CITATION_WEIGHT * breakdown.citation_accuracy
    )
    assert breakdown.kind == "answer"
    assert breakdown.composite == pytest.approx(expected)


def test_as_refusal_recasts_a_suppressed_answer_and_is_idempotent():
    """A below-threshold answer is a refusal too, and reports like one."""
    src = chunk()
    src.rerank_score = -9.0
    weak = GeneratedAnswer("q", "Claim [1].", parse_claims("A shaky claim here [1]."), [src])

    answered = score_confidence(weak, [src], verify(weak, StubLLM(mode="judge")))
    assert answered.kind == "answer"

    recast = answered.as_refusal()
    assert recast.kind == "refusal"
    assert recast.coverage is None
    assert recast.composite == pytest.approx(1.0 - answered.retrieval)
    assert recast.as_refusal() is recast
