from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import EmbeddingConfig


class Embedder:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = SentenceTransformer(config.model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


@lru_cache
def get_embedder(model_name: str, batch_size: int) -> Embedder:
    return Embedder(EmbeddingConfig(model_name=model_name, batch_size=batch_size))
