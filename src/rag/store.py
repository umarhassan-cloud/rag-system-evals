from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from rag.config import AppConfig
from rag.models import DocumentChunk


class VectorStore:
    def __init__(self, config: AppConfig):
        self.config = config
        persist_dir = Path(config.paths.chroma_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection: Collection = self._client.get_or_create_collection(
            name=config.paths.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        name = self.config.paths.collection_name
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
        self.collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas: list[dict[str, Any]] = []
        for c in chunks:
            meta = {
                "doc_id": c.doc_id,
                "source": c.source,
                **{k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))},
            }
            metadatas.append(meta)

        # Chroma has practical batch limits; upsert in slices.
        batch_size = 200
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
        return len(chunks)

    def count(self) -> int:
        return int(self.collection.count())

    def get_all_chunks(self) -> list[DocumentChunk]:
        total = self.count()
        if total == 0:
            return []
        result = self.collection.get(include=["documents", "metadatas"])
        chunks: list[DocumentChunk] = []
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        for chunk_id, text, meta in zip(ids, docs, metas, strict=False):
            meta = meta or {}
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=str(meta.get("doc_id", "")),
                    source=str(meta.get("source", "")),
                    text=text or "",
                    metadata={k: v for k, v in meta.items() if k not in {"doc_id", "source"}},
                )
            )
        return chunks

    def query_dense(self, embedding: list[float], top_k: int) -> list[tuple[DocumentChunk, float]]:
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        scored: list[tuple[DocumentChunk, float]] = []
        for chunk_id, text, meta, distance in zip(ids, docs, metas, distances, strict=False):
            meta = meta or {}
            # Cosine distance -> similarity
            score = 1.0 - float(distance)
            scored.append(
                (
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=str(meta.get("doc_id", "")),
                        source=str(meta.get("source", "")),
                        text=text or "",
                        metadata={k: v for k, v in meta.items() if k not in {"doc_id", "source"}},
                    ),
                    score,
                )
            )
        return scored
