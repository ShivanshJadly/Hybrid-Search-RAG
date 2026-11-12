from __future__ import annotations

import hashlib
import re
from pathlib import Path
from threading import Event, Thread

import numpy as np
import pytest
from fastapi.testclient import TestClient

import rag.pipeline as pipeline_module
from rag.config import Settings
from rag.loaders.registry import source_fingerprint
from rag.models import Chunk, ChunkMetadata, LoadedDocument, make_doc_id
from rag.pipeline import ingest
from rag.stores.bm25 import Bm25Store
from rag.stores.documents import DocumentStore
from rag.stores.vector import VectorStore

_WORD = re.compile(r"\S+")
_STRATEGIES = ("fixed", "header", "semantic")


class FakeEmbedder:
    def __init__(self):
        self.embedded: list[str] = []

    def validate(self):
        pass

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in _WORD.finditer(text)]

    def count(self, text: str) -> int:
        return len(self.offsets(text))

    def embed_passages(
        self, texts: list[str], show_progress: bool = False
    ) -> np.ndarray:
        self.embedded.extend(texts)
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for index, text in enumerate(texts):
            slot = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 384
            vectors[index, slot] = 1.0
        return vectors


class PausingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.old_version_started = Event()
        self.new_version_started = Event()
        self.release_old_version = Event()

    def embed_passages(
        self, texts: list[str], show_progress: bool = False
    ) -> np.ndarray:
        combined = "\n".join(texts)
        if "Version=old" in combined:
            self.old_version_started.set()
            if not self.release_old_version.wait(timeout=10):
                raise TimeoutError("old-version embedding was not released")
        if "Version=new" in combined:
            self.new_version_started.set()
        return super().embed_passages(texts, show_progress)


def _settings(tmp_path: Path) -> Settings:
    root = tmp_path / "documents"
    root.mkdir(parents=True, exist_ok=True)
    return Settings(
        data_dir=root,
        storage_dir=tmp_path / "storage",
        chunk_tokens=1000,
        chunk_overlap_tokens=0,
        min_chunk_chars=80,
    )


def _write_document(root: Path, text: str, name: str = "guide.md") -> Path:
    path = root / name
    path.write_text(text, encoding="utf-8")
    return path


def _vector_records(settings: Settings, strategy: str, source: str) -> dict:
    store = VectorStore(settings.chroma_dir, strategy)
    return store.collection.get(where={"source": source}, include=["metadatas"])


def _seed_index(
    settings: Settings,
    doc_id: str,
    source: str,
    strategy: str,
    *,
    legacy_metadata: bool = False,
) -> None:
    chunk_id = f"{strategy}-chunk"
    text = f"Stored searchable text for {source} under {strategy}."
    meta = ChunkMetadata(
        source=source,
        source_name=Path(source).name,
        media_type="md",
        strategy=strategy,
        ordinal=0,
        char_start=0,
        char_end=len(text),
        n_tokens=len(text.split()),
    )
    chunk = Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text, meta=meta)

    vector = VectorStore(settings.chroma_dir, strategy)
    vector.add([chunk], np.ones((1, 384), dtype=np.float32))
    if legacy_metadata:
        vector.collection.delete(ids=[chunk_id])
        vector.collection.upsert(
            ids=[chunk_id],
            documents=[text],
            embeddings=[np.ones(384, dtype=np.float32).tolist()],
            metadatas=[meta.as_dict()],
        )

    bm25 = Bm25Store(settings.bm25_dir, strategy)
    bm25.add([chunk])
    if legacy_metadata:
        bm25.metas[0].pop("doc_id", None)
    bm25.save()


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    from rag.api import app as api_module

    settings = _settings(tmp_path)
    settings.ensure_dirs()
    monkeypatch.setattr(api_module, "settings", settings)
    client = TestClient(api_module.app, raise_server_exceptions=False)
    return client, settings, api_module


def test_modified_reingest_removes_old_ids_from_both_indexes(tmp_path):
    settings = _settings(tmp_path)
    embedder = FakeEmbedder()
    old_text = "# Deploy guide\n\nSet rollout_flag=legacy before starting the service. " + "Details. " * 20
    path = _write_document(settings.data_dir, old_text)

    first = ingest(settings.data_dir, settings, embedder, "fixed")
    source = path.relative_to(settings.data_dir).as_posix()
    doc_id = make_doc_id(source)
    old_vector_id = _vector_records(settings, "fixed", source)["ids"][0]
    bm25 = Bm25Store(settings.bm25_dir, "fixed")
    assert bm25.load()
    old_bm25_id = bm25.chunk_ids[0]
    assert first.vector_count == first.bm25_count == 1

    # Existing persisted rows have source metadata but predate doc_id metadata.
    for metadata in bm25.metas:
        metadata.pop("doc_id", None)
    bm25.save()

    path.write_text(old_text.replace("rollout_flag=legacy", "rollout_flag=current"), encoding="utf-8")
    second = ingest(settings.data_dir, settings, embedder, "fixed")

    vector_records = _vector_records(settings, "fixed", source)
    bm25 = Bm25Store(settings.bm25_dir, "fixed")
    assert bm25.load()
    assert len(vector_records["ids"]) == bm25.count() == 1
    assert old_vector_id not in vector_records["ids"]
    assert old_bm25_id not in bm25.chunk_ids
    assert vector_records["metadatas"][0]["doc_id"] == doc_id
    assert second.in_sync


def test_unchanged_source_skips_loader_chunker_and_embedding(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    embedder = FakeEmbedder()
    _write_document(settings.data_dir, "# Guide\n\n" + "A stable searchable sentence. " * 20)
    loader = pipeline_module.load_file
    chunker = pipeline_module.build_chunks
    calls = {"load": 0, "chunk": 0}

    def counted_load(*args, **kwargs):
        calls["load"] += 1
        return loader(*args, **kwargs)

    def counted_chunk(*args, **kwargs):
        calls["chunk"] += 1
        return chunker(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "load_file", counted_load)
    monkeypatch.setattr(pipeline_module, "build_chunks", counted_chunk)
    ingest(settings.data_dir, settings, embedder, "fixed")
    first_counts = (calls["load"], calls["chunk"], len(embedder.embedded))

    report = ingest(settings.data_dir, settings, embedder, "fixed")

    assert report.files_unchanged == 1
    assert report.files_loaded == 0
    assert (calls["load"], calls["chunk"], len(embedder.embedded)) == first_counts
    assert report.vector_count == report.bm25_count == 1


def test_fingerprints_are_per_strategy_and_rebuild_invalidates_them(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path)
    embedder = FakeEmbedder()
    path = _write_document(settings.data_dir, "# Guide\n\n" + "Shared stable text. " * 20)

    ingest(settings.data_dir, settings, embedder, "fixed")
    first_embed_count = len(embedder.embedded)
    header = ingest(settings.data_dir, settings, embedder, "header")
    assert header.files_loaded == 1
    assert len(embedder.embedded) > first_embed_count

    fixed_again = ingest(settings.data_dir, settings, embedder, "fixed")
    header_again = ingest(settings.data_dir, settings, embedder, "header")
    assert fixed_again.files_unchanged == header_again.files_unchanged == 1

    load_file = pipeline_module.load_file

    def fail_load(*args, **kwargs):
        raise OSError("temporary read failure")

    monkeypatch.setattr(pipeline_module, "load_file", fail_load)
    rebuilt = ingest(settings.data_dir, settings, embedder, "fixed", rebuild=True)
    doc_id = make_doc_id(path.name)
    assert rebuilt.files_failed
    assert DocumentStore(settings.documents_dir).content_hash(doc_id, "fixed") is None
    assert VectorStore(settings.chroma_dir, "fixed").count() == 0

    monkeypatch.setattr(pipeline_module, "load_file", load_file)
    recovered = ingest(settings.data_dir, settings, embedder, "fixed")
    assert recovered.files_loaded == 1
    assert recovered.vector_count == recovered.bm25_count == 1


def test_reingest_removes_chunks_when_document_becomes_empty(tmp_path):
    settings = _settings(tmp_path)
    embedder = FakeEmbedder()
    path = _write_document(settings.data_dir, "# Guide\n\n" + "Searchable content. " * 20)
    ingest(settings.data_dir, settings, embedder, "fixed")
    path.write_text("   \n", encoding="utf-8")

    report = ingest(settings.data_dir, settings, embedder, "fixed")

    assert report.vector_count == report.bm25_count == 0
    assert VectorStore(settings.chroma_dir, "fixed").count() == 0
    bm25 = Bm25Store(settings.bm25_dir, "fixed")
    assert bm25.load()
    assert bm25.count() == 0
    assert DocumentStore(settings.documents_dir).load(make_doc_id(path.name)).text == ""


def test_reingest_recovers_after_partial_index_write(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    embedder = FakeEmbedder()
    text = "# Guide\n\n" + "Version=old searchable body. " * 20
    path = _write_document(settings.data_dir, text)
    ingest(settings.data_dir, settings, embedder, "fixed")
    path.write_text(text.replace("Version=old", "Version=new"), encoding="utf-8")

    add = Bm25Store.add
    failed = False

    def fail_once(store, chunks):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("simulated BM25 write failure")
        return add(store, chunks)

    monkeypatch.setattr(Bm25Store, "add", fail_once)
    with pytest.raises(OSError, match="simulated BM25"):
        ingest(settings.data_dir, settings, embedder, "fixed")

    monkeypatch.setattr(Bm25Store, "add", add)
    report = ingest(settings.data_dir, settings, embedder, "fixed")

    source = path.relative_to(settings.data_dir).as_posix()
    vector_records = _vector_records(settings, "fixed", source)
    bm25 = Bm25Store(settings.bm25_dir, "fixed")
    assert bm25.load()
    assert len(vector_records["ids"]) == bm25.count() == 1
    assert report.in_sync


def test_overlapping_background_ingests_leave_the_latest_source_version(tmp_path):
    settings = _settings(tmp_path)
    embedder = PausingEmbedder()
    original = "# Guide\n\nVersion=old searchable body. " + "Details. " * 20
    path = _write_document(settings.data_dir, original)
    errors: list[Exception] = []

    def run_ingest():
        try:
            ingest(settings.data_dir, settings, embedder, "fixed")
        except Exception as exc:
            errors.append(exc)

    first = Thread(target=run_ingest, daemon=True)
    second_started = Event()
    second: Thread | None = None
    overlapped = False
    first.start()
    try:
        assert embedder.old_version_started.wait(timeout=10)
        path.write_text(
            original.replace("Version=old", "Version=new"), encoding="utf-8"
        )

        def run_second_ingest():
            second_started.set()
            run_ingest()

        second = Thread(target=run_second_ingest, daemon=True)
        second.start()
        assert second_started.wait(timeout=10)
        overlapped = embedder.new_version_started.wait(timeout=1)
    finally:
        embedder.release_old_version.set()
        first.join(timeout=10)
        if second is not None:
            second.join(timeout=10)

    assert not first.is_alive()
    assert second is not None and not second.is_alive()
    assert not errors
    assert not overlapped

    source = path.relative_to(settings.data_dir).as_posix()
    vector_records = VectorStore(settings.chroma_dir, "fixed").collection.get(
        where={"source": source}, include=["documents", "metadatas"]
    )
    bm25 = Bm25Store(settings.bm25_dir, "fixed")
    assert bm25.load()
    assert len(vector_records["ids"]) == bm25.count() == 1
    assert "Version=new" in vector_records["documents"][0]
    assert "Version=new" in bm25.texts[0]


def test_delete_api_removes_document_from_every_strategy(api_client):
    client, settings, _ = api_client
    source = "guide.md"
    doc_id = make_doc_id(source)
    DocumentStore(settings.documents_dir).save(
        LoadedDocument(
            doc_id=doc_id,
            source=source,
            source_name="guide.md",
            media_type="md",
            text="Stored source text.",
        )
    )
    for strategy in _STRATEGIES:
        _seed_index(settings, doc_id, source, strategy, legacy_metadata=True)

    response = client.delete(f"/v1/documents/{doc_id}")

    assert response.status_code == 200
    assert response.json() == {"status": "deleted", "doc_id": doc_id}
    assert DocumentStore(settings.documents_dir).load(doc_id) is None
    for strategy in _STRATEGIES:
        assert VectorStore(settings.chroma_dir, strategy).count() == 0
        bm25 = Bm25Store(settings.bm25_dir, strategy)
        assert bm25.load()
        assert bm25.count() == 0


def test_delete_api_retry_finishes_partial_deletion(api_client, monkeypatch):
    client, settings, api_module = api_client
    source = "guide.md"
    doc_id = make_doc_id(source)
    DocumentStore(settings.documents_dir).save(
        LoadedDocument(
            doc_id=doc_id,
            source=source,
            source_name="guide.md",
            media_type="md",
            text="Stored source text.",
        )
    )
    for strategy in _STRATEGIES:
        _seed_index(settings, doc_id, source, strategy)

    delete = Bm25Store.delete
    failed = False

    def fail_once(store, target_id, source=None):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("simulated delete failure")
        return delete(store, target_id, source)

    monkeypatch.setattr(api_module.Bm25Store, "delete", fail_once)
    first = client.delete(f"/v1/documents/{doc_id}")
    assert first.status_code == 500
    assert DocumentStore(settings.documents_dir).load(doc_id) is not None

    monkeypatch.setattr(api_module.Bm25Store, "delete", delete)
    second = client.delete(f"/v1/documents/{doc_id}")
    assert second.status_code == 200
    assert DocumentStore(settings.documents_dir).load(doc_id) is None
    for strategy in _STRATEGIES:
        assert VectorStore(settings.chroma_dir, strategy).count() == 0
        bm25 = Bm25Store(settings.bm25_dir, strategy)
        assert bm25.load()
        assert bm25.count() == 0


def test_delete_api_can_remove_orphaned_legacy_index_rows(api_client):
    client, settings, _ = api_client
    source = "guide.md"
    doc_id = make_doc_id(source)
    for strategy in _STRATEGIES:
        _seed_index(settings, doc_id, source, strategy, legacy_metadata=True)

    response = client.delete(f"/v1/documents/{doc_id}")

    assert response.status_code == 200
    for strategy in _STRATEGIES:
        assert VectorStore(settings.chroma_dir, strategy).count() == 0
        bm25 = Bm25Store(settings.bm25_dir, strategy)
        assert bm25.load()
        assert bm25.count() == 0


def test_delete_api_rejects_invalid_document_id(api_client):
    client, _, _ = api_client
    assert client.delete("/v1/documents/not-a-document-id").status_code == 404


def test_source_fingerprint_is_exact_not_normalized(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("A Case-Sensitive Flag", encoding="utf-8")
    first = source_fingerprint(path)
    path.write_text("a case-sensitive flag ", encoding="utf-8")

    assert source_fingerprint(path) != first
