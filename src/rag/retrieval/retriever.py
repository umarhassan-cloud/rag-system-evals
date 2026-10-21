from __future__ import annotations

from functools import lru_cache
from typing import Literal

from sentence_transformers import CrossEncoder

from rag.config import AppConfig
from rag.embeddings import Embedder, get_embedder
from rag.models import RetrievedChunk
from rag.retrieval import attach_ranks
from rag.retrieval.bm25 import BM25Index, reciprocal_rank_fusion
from rag.store import VectorStore

StrategyName = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


@lru_cache
def get_cross_encoder(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


class Retriever:
    def __init__(self, config: AppConfig, store: VectorStore, embedder: Embedder | None = None):
        self.config = config
        self.store = store
        self.embedder = embedder or get_embedder(
            config.embedding.model_name,
            config.embedding.batch_size,
        )
        self.bm25 = BM25Index(store.get_all_chunks())

    def refresh_bm25(self) -> None:
        self.bm25.build(self.store.get_all_chunks())

    def retrieve(self, query: str, strategy: StrategyName = "hybrid", top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.config.retrieval.top_k
        if strategy == "dense":
            return self._dense(query, k)
        if strategy == "bm25":
            return self.bm25.search(query, k)
        if strategy == "hybrid":
            return self._hybrid(query, k)
        if strategy == "hybrid_rerank":
            return self._hybrid_rerank(query, k)
        raise ValueError(f"Unknown strategy: {strategy}")

    def _dense(self, query: str, top_k: int) -> list[RetrievedChunk]:
        vector = self.embedder.embed_query(query).tolist()
        pairs = self.store.query_dense(vector, top_k)
        return attach_ranks(pairs, "dense")

    def _hybrid(self, query: str, top_k: int) -> list[RetrievedChunk]:
        candidate_k = max(top_k, self.config.retrieval.rerank_candidates)
        dense = self._dense(query, candidate_k)
        lexical = self.bm25.search(query, candidate_k)
        return reciprocal_rank_fusion(
            [dense, lexical],
            rrf_k=self.config.retrieval.rrf_k,
            top_k=top_k,
            strategy="hybrid",
        )

    def _hybrid_rerank(self, query: str, top_k: int) -> list[RetrievedChunk]:
        candidates = self._hybrid(query, self.config.retrieval.rerank_candidates)
        if not candidates:
            return []
        model = get_cross_encoder(self.config.retrieval.rerank_model)
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = model.predict(pairs)
        rescored = sorted(
            zip(candidates, scores, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]
        return [
            RetrievedChunk(
                chunk=item.chunk,
                score=float(score),
                rank=i,
                strategy="hybrid_rerank",
            )
            for i, (item, score) in enumerate(rescored, start=1)
        ]
