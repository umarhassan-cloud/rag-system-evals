from __future__ import annotations

from pathlib import Path

from rag.config import AppConfig, get_config
from rag.embeddings import get_embedder
from rag.generate import Generator
from rag.ingest import ingest_paths
from rag.models import AnswerResult
from rag.retrieval.retriever import Retriever, StrategyName
from rag.store import VectorStore


class RAGPipeline:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or get_config()
        self.store = VectorStore(self.config)
        self.embedder = get_embedder(
            self.config.embedding.model_name,
            self.config.embedding.batch_size,
        )
        self.retriever = Retriever(self.config, self.store, self.embedder)
        self.generator = Generator(self.config)

    def ingest(self, paths: list[str | Path], *, reset: bool = False) -> dict[str, int]:
        resolved = [Path(p) for p in paths]
        chunks = ingest_paths(resolved, self.config.chunking)
        if reset:
            self.store.reset()
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        count = self.store.upsert_chunks(chunks, embeddings.tolist())
        self.retriever.refresh_bm25()
        return {"chunks": count, "collection_size": self.store.count()}

    def ask(self, question: str, strategy: StrategyName = "hybrid", top_k: int | None = None) -> AnswerResult:
        retrieved = self.retriever.retrieve(question, strategy=strategy, top_k=top_k)
        return self.generator.answer(question, retrieved, strategy=strategy)
