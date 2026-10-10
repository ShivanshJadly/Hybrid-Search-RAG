"""Local embedding model.

Runs on CPU. Note for this machine specifically: PyTorch has no ROCm build for
Windows, so the Radeon GPU is not used here. That is fine -- bge-small is 33M
parameters and embeds a few thousand chunks in well under a minute. The GPU
does get used later, by Ollama, which reaches the card through Vulkan rather
than PyTorch.
"""

from __future__ import annotations

import logging
from functools import cached_property

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Settings, validate_chunk_size

log = logging.getLogger(__name__)


class LocalEmbedder:
    """Wraps SentenceTransformer and exposes the tokenizer the chunkers need."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            log.info("Loading embedding model %s", self.settings.embedding_model)
            self._model = SentenceTransformer(
                self.settings.embedding_model, device="cpu"
            )
        return self._model

    @cached_property
    def max_seq_length(self) -> int:
        return int(self.model.max_seq_length)

    @cached_property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def validate(self) -> None:
        """Fail loudly at startup rather than silently truncating later."""
        validate_chunk_size(self.settings.chunk_tokens, self.max_seq_length)

    # --- Tokenizer surface (satisfies chunking.base.Tokenizer) ---

    def offsets(self, text: str) -> list[tuple[int, int]]:
        enc = self.model.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        # Zero-width offsets come from special/control tokens; drop them.
        return [(s, e) for s, e in enc["offset_mapping"] if e > s]

    def count(self, text: str) -> int:
        return len(
            self.model.tokenizer(
                text, add_special_tokens=False, truncation=False, verbose=False
            )["input_ids"]
        )

    # --- Encoding ---

    def embed_passages(
        self, texts: list[str], show_progress: bool = False
    ) -> np.ndarray:
        """Documents are embedded bare -- bge applies no prefix to passages."""
        return self.model.encode(
            texts,
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )

    def embed_query(self, text: str) -> np.ndarray:
        """Queries get the bge asymmetric prefix. Omitting it measurably hurts
        retrieval, and it is an easy thing to forget on the query path only."""
        return self.model.encode(
            [self.settings.query_prefix + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
