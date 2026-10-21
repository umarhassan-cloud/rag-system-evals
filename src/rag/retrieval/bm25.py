from __future__ import annotations

import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from rag.models import DocumentChunk, RetrievedChunk
from rag.retrieval import attach_ranks


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, chunks: list[DocumentChunk] | None = None):
        self.chunks: list[DocumentChunk] = []
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] = []
        if chunks:
            self.build(chunks)

    def build(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self._corpus_tokens = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self.chunks else None

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not self._bm25 or not self.chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        pairs = [(self.chunks[i], float(scores[i])) for i in ranked_idx if scores[i] > 0]
        if not pairs:
            # Fall back to top scores even if all zero so evals still get candidates.
            pairs = [(self.chunks[i], float(scores[i])) for i in ranked_idx]
        return attach_ranks(pairs, "bm25")


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    *,
    rrf_k: int = 60,
    top_k: int = 5,
    strategy: str = "hybrid",
) -> list[RetrievedChunk]:
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for item in ranked:
            scores[item.chunk.chunk_id] += 1.0 / (rrf_k + item.rank)
            prev = best.get(item.chunk.chunk_id)
            if prev is None or item.score > prev.score:
                best[item.chunk.chunk_id] = item

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    fused: list[RetrievedChunk] = []
    for rank, (chunk_id, score) in enumerate(ordered, start=1):
        base = best[chunk_id]
        fused.append(
            RetrievedChunk(
                chunk=base.chunk,
                score=float(score),
                rank=rank,
                strategy=strategy,
            )
        )
    return fused
