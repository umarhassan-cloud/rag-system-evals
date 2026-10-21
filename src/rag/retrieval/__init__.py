from __future__ import annotations

from rag.models import RetrievedChunk

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")


def attach_ranks(items: list[tuple], strategy: str) -> list[RetrievedChunk]:
    """Convert (DocumentChunk, score) pairs into ranked RetrievedChunk list."""
    ranked: list[RetrievedChunk] = []
    for i, (chunk, score) in enumerate(items, start=1):
        ranked.append(RetrievedChunk(chunk=chunk, score=float(score), rank=i, strategy=strategy))
    return ranked
