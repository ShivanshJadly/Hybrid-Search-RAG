from __future__ import annotations

import re
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

import rag.api.app as api_module
import rag.stores.bm25 as bm25_module
from rag.answering import AnswerPipeline
from rag.config import Settings
from rag.llm.stub import StubLLM
from rag.models import Chunk, ChunkMetadata
from rag.rerank import NoOpReranker
from rag.stores.bm25 import Bm25Store


def make_chunk(chunk_id: str, marker: str) -> Chunk:
    text = f"Operational runbook exact marker {marker} controls deployment recovery."
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=text,
        meta=ChunkMetadata(
            source=f"{chunk_id}.txt",
            source_name=f"{chunk_id}.txt",
            media_type="txt",
            strategy="fixed",
            ordinal=0,
            char_start=0,
            char_end=len(text),
            n_tokens=len(text.split()),
        ),
    )


class FakeEmbedder:
    markers = ("qxvnebula73019", "qxvharbor42083", "qxvforest61527")

    def validate(self) -> None:
        pass

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]

    def count(self, text: str) -> int:
        return len(self.offsets(text))

    def _vector(self, text: str) -> np.ndarray:
        marker = next((item for item in self.markers if item in text.lower()), None)
        if marker is None:
            return np.ones(len(self.markers), dtype=np.float32)
        return np.eye(len(self.markers), dtype=np.float32)[self.markers.index(marker)]

    def embed_passages(
        self, texts: list[str], show_progress: bool = False
    ) -> np.ndarray:
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


def test_api_ingest_refreshes_the_running_sparse_retriever(tmp_path, monkeypatch):
    root = tmp_path / "docs"
    root.mkdir()
    for index, marker in enumerate(FakeEmbedder.markers):
        text = (
            "The production deployment runbook explains rollout and rollback. "
            f"The exact lookup marker {marker} identifies the matching controls."
        )
        (root / f"guide-{index}.txt").write_text(text, encoding="utf-8")

    settings = Settings(
        _env_file=None,
        data_dir=root,
        storage_dir=tmp_path / "storage",
        min_chunk_chars=20,
        rerank_enabled=False,
    )
    embedder = FakeEmbedder()
    llm = StubLLM()
    judge = StubLLM(mode="judge")
    pipeline = AnswerPipeline(
        settings,
        embedder,
        llm,
        judge,
        strategy="fixed",
        reranker=NoOpReranker(),
    )
    monkeypatch.setattr(api_module, "settings", settings)
    monkeypatch.setattr(
        api_module,
        "_state",
        {
            "pipelines": {"fixed": pipeline},
            "embedder": embedder,
            "llm": llm,
            "judge": judge,
        },
    )
    client = TestClient(api_module.app, raise_server_exceptions=False)

    ingest = client.post("/v1/ingest", json={"path": str(root), "strategy": "fixed"})
    answer = client.post(
        "/v1/ask",
        json={
            "question": FakeEmbedder.markers[0],
            "strategy": "fixed",
            "top_k": 5,
            "use_dense": True,
            "use_sparse": True,
            "rerank": False,
            "verify_citations": False,
        },
    )

    assert ingest.status_code == 200
    assert answer.status_code == 200
    payload = answer.json()
    assert payload["retrieval"]["sparse_hits"] > 0
    assert any(
        source["source"] == "guide-0.txt"
        and source["found_by"] in {"sparse", "both"}
        for source in payload["sources"]
    )


def test_query_reloads_snapshot_saved_by_another_store(tmp_path):
    writer = Bm25Store(tmp_path, "fixed")
    writer.add(
        [
            make_chunk("first", "qxvnebula73019"),
            make_chunk("filler-1", "qxvfiller90127"),
            make_chunk("filler-2", "qxvfiller84316"),
        ]
    )
    writer.save()

    reader = Bm25Store(tmp_path, "fixed")
    assert reader.load()
    assert reader.query("qxvnebula73019", 5)[0]["chunk_id"] == "first"

    writer = Bm25Store(tmp_path, "fixed")
    assert writer.load()
    writer.add([make_chunk("second", "qxvharbor42083")])
    writer.save()

    result = reader.query("qxvharbor42083", 5)
    assert [item["chunk_id"] for item in result] == ["second"]


def test_query_clears_snapshot_after_persisted_index_is_removed(tmp_path):
    writer = Bm25Store(tmp_path, "fixed")
    writer.add(
        [
            make_chunk("first", "qxvnebula73019"),
            make_chunk("filler-1", "qxvfiller90127"),
            make_chunk("filler-2", "qxvfiller84316"),
        ]
    )
    writer.save()

    reader = Bm25Store(tmp_path, "fixed")
    assert reader.load()
    assert reader.query("qxvnebula73019", 5)

    writer.reset()

    assert reader.query("qxvnebula73019", 5) == []
    assert reader.count() == 0


def test_failed_atomic_save_keeps_last_valid_snapshot_and_retry_refreshes(
    tmp_path, monkeypatch
):
    writer = Bm25Store(tmp_path, "fixed")
    writer.add(
        [
            make_chunk("first", "qxvnebula73019"),
            make_chunk("filler-1", "qxvfiller90127"),
            make_chunk("filler-2", "qxvfiller84316"),
        ]
    )
    writer.save()
    previous_bytes = writer.path.read_bytes()

    reader = Bm25Store(tmp_path, "fixed")
    assert reader.load()

    writer.add([make_chunk("second", "qxvharbor42083")])

    def fail_replace(source, destination):
        raise PermissionError("simulated replace failure")

    monkeypatch.setattr(bm25_module.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="simulated replace failure"):
        writer.save()

    assert writer.path.read_bytes() == previous_bytes
    assert [item["chunk_id"] for item in reader.query("qxvnebula73019", 5)] == [
        "first"
    ]
    assert reader.query("qxvharbor42083", 5) == []

    monkeypatch.undo()
    writer.save()
    assert [
        item["chunk_id"] for item in reader.query("qxvharbor42083", 5)
    ] == ["second"]


def test_save_does_not_mark_a_concurrent_last_writer_as_its_own_snapshot(
    tmp_path, monkeypatch
):
    shared_dir = tmp_path / "shared"
    writer = Bm25Store(shared_dir, "fixed")
    writer.add(
        [
            make_chunk("first", "qxvnebula73019"),
            make_chunk("filler-1", "qxvfiller90127"),
            make_chunk("filler-2", "qxvfiller84316"),
        ]
    )
    writer.save()
    assert writer.load()
    writer.add([make_chunk("writer", "qxvwriter27591")])

    concurrent = Bm25Store(tmp_path / "concurrent", "fixed")
    concurrent.add(
        [
            make_chunk("latest", "qxvlatest68143"),
            make_chunk("filler-1", "qxvfiller90127"),
            make_chunk("filler-2", "qxvfiller84316"),
        ]
    )
    concurrent.save()

    replace = bm25_module.os.replace

    def replace_then_concurrent_write(source, destination):
        replace(source, destination)
        replace(concurrent.path, destination)

    monkeypatch.setattr(bm25_module.os, "replace", replace_then_concurrent_write)
    writer.save()

    result = writer.query("qxvlatest68143", 5)
    assert [item["chunk_id"] for item in result] == ["latest"]


def test_save_retries_a_transient_windows_sharing_violation(tmp_path, monkeypatch):
    class SharingViolation(PermissionError):
        @property
        def winerror(self) -> int:
            return 32

    writer = Bm25Store(tmp_path, "fixed")
    writer.add(
        [
            make_chunk("first", "qxvnebula73019"),
            make_chunk("filler-1", "qxvfiller90127"),
            make_chunk("filler-2", "qxvfiller84316"),
        ]
    )
    replace = bm25_module.os.replace
    attempts = 0

    def fail_once(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise SharingViolation("simulated sharing violation")
        replace(source, destination)

    monkeypatch.setattr(bm25_module.os, "replace", fail_once)
    writer.save()

    assert attempts == 2
    reader = Bm25Store(tmp_path, "fixed")
    assert reader.load()
    assert [item["chunk_id"] for item in reader.query("qxvnebula73019", 5)] == [
        "first"
    ]


def test_queries_observe_only_complete_snapshots_during_repeated_replacements(
    tmp_path,
):
    first = make_chunk("first", "qxvnebula73019")
    second = make_chunk("second", "qxvharbor42083")
    fillers = [
        make_chunk("filler-1", "qxvfiller90127"),
        make_chunk("filler-2", "qxvfiller84316"),
    ]
    writer = Bm25Store(tmp_path, "fixed")
    writer.add([first, *fillers])
    writer.save()
    reader = Bm25Store(tmp_path, "fixed")
    assert reader.load()

    start = threading.Barrier(2)
    errors: list[BaseException] = []

    def replace_snapshots() -> None:
        try:
            start.wait()
            for index in range(30):
                current = Bm25Store(tmp_path, "fixed")
                current.add([first if index % 2 else second, *fillers])
                current.save()
        except BaseException as exc:
            errors.append(exc)

    def read_snapshots() -> None:
        try:
            start.wait()
            for _ in range(100):
                for marker in ("qxvnebula73019", "qxvharbor42083"):
                    for result in reader.query(marker, 5):
                        assert result["chunk_id"] in {"first", "second"}
                        assert result["text"] in {first.text, second.text}
        except BaseException as exc:
            errors.append(exc)

    writer_thread = threading.Thread(target=replace_snapshots)
    reader_thread = threading.Thread(target=read_snapshots)
    writer_thread.start()
    reader_thread.start()
    writer_thread.join()
    reader_thread.join()

    assert errors == [], [
        (type(error).__name__, error.errno, getattr(error, "winerror", None), str(error))
        for error in errors
    ]
