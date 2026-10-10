"""Central configuration.

Every tunable that Phase 4 will want to sweep lives here, not inline in the
code. Chunk sizes in particular are deliberately *guesses* right now -- the
evaluation phase replaces them with measured values.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="RAG_", extra="ignore"
    )

    # --- Paths ---
    data_dir: Path = PROJECT_ROOT / "data" / "raw"
    storage_dir: Path = PROJECT_ROOT / "storage"

    # --- Embeddings ---
    # bge-small-en-v1.5: 33M params, 384-dim, 512-token ceiling.
    # Small enough to run comfortably on CPU.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    # bge was trained with an asymmetric query prefix; passages get none.
    query_prefix: str = "Represent this sentence for searching relevant passages: "

    # --- Chunking ---
    # Kept under the embedding model's 512-token limit with headroom for the
    # special tokens the tokenizer adds. See validate_chunk_size().
    chunk_tokens: int = 400
    chunk_overlap_tokens: int = 60
    min_chunk_chars: int = 80  # drop slivers left over at document tails

    # header strategy: merge adjacent short sections that fit in one chunk.
    # Trades a little attribution precision (the merged chunk cites the first
    # section's heading) for chunks big enough to be worth retrieving.
    header_merge_small: bool = True

    # semantic strategy: a boundary is placed where the distance between
    # consecutive sentences lands at or above this percentile for the document.
    semantic_percentile: float = 95.0
    semantic_buffer_size: int = 1  # neighbours included when embedding a sentence

    # --- Deduplication ---
    # Cosine similarity at or above which two chunks are treated as duplicates.
    near_duplicate_threshold: float = 0.95

    # --- Retrieval (Phase 2) ---
    dense_top_k: int = 10
    sparse_top_k: int = 10

    # Reciprocal Rank Fusion. The constant damps the influence of top ranks;
    # 60 is the value from the original RRF paper and is a sane default.
    rrf_k: int = 60
    dense_weight: float = 0.7
    sparse_weight: float = 0.3

    # Reranking: pull a wide candidate set, then let a cross-encoder pick.
    rerank_enabled: bool = True
    rerank_candidates: int = 20
    rerank_top_n: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Generation (Phase 3) ---
    llm_backend: str = "gemini"  # gemini | groq | ollama | stub
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    groq_model: str = "llama-3.3-70b-versatile"
    # Gemini's model lineup moves fast and free-tier availability is not
    # published. Run `python -m rag check` to list what your key can actually
    # reach, then set RAG_GEMINI_MODEL if this default 404s.
    gemini_model: str = "gemini-3.5-flash"
    llm_timeout_seconds: int = 120

    # API keys. Accepted under either the RAG_ prefix or the bare provider
    # name, because both spellings are natural and only one of them working
    # is an unpleasant thing to debug.
    groq_api_key: str = Field(
        "", validation_alias=AliasChoices("RAG_GROQ_API_KEY", "GROQ_API_KEY")
    )
    gemini_api_key: str = Field(
        "", validation_alias=AliasChoices("RAG_GEMINI_API_KEY", "GEMINI_API_KEY")
    )

    # Judge backend, used for citation verification and evaluation. Kept
    # separate from the generator so the grader can differ from the writer
    # (Notion plan, Appendix A.4) -- and so a cheaper model can grade.
    judge_backend: str = "gemini"
    judge_model: str = ""  # blank means "same model as the backend default"

    # Below this composite confidence, the system says it does not know
    # rather than answering from weak context.
    min_answer_confidence: float = 0.35

    @property
    def chroma_dir(self) -> Path:
        return self.storage_dir / "chroma"

    @property
    def bm25_dir(self) -> Path:
        return self.storage_dir / "bm25"

    @property
    def documents_dir(self) -> Path:
        return self.storage_dir / "documents"

    def ensure_dirs(self) -> None:
        for p in (
            self.data_dir,
            self.storage_dir,
            self.chroma_dir,
            self.bm25_dir,
            self.documents_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()


def validate_chunk_size(chunk_tokens: int, model_max_tokens: int) -> None:
    """Guard against the quietest failure mode in this whole pipeline.

    Sentence-transformers silently truncates anything past the model's max
    sequence length. A chunk larger than the limit still embeds, still stores,
    still retrieves -- it is just missing its tail, with no error anywhere.
    """
    budget = model_max_tokens - 8  # leave room for [CLS]/[SEP] and friends
    if chunk_tokens > budget:
        raise ValueError(
            f"chunk_tokens={chunk_tokens} exceeds the safe budget of {budget} "
            f"for a model with max_seq_length={model_max_tokens}. Embeddings "
            f"would be silently truncated. Lower chunk_tokens in config.py."
        )
