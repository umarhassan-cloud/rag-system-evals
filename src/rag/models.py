from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    rank: int
    strategy: str

    @property
    def citation_label(self) -> str:
        return f"[{self.rank}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "strategy": self.strategy,
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "source": self.chunk.source,
            "text": self.chunk.text,
            "metadata": self.chunk.metadata,
        }


@dataclass(slots=True)
class Citation:
    index: int
    source: str
    chunk_id: str
    doc_id: str
    excerpt: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnswerResult:
    question: str
    answer: str
    strategy: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "strategy": self.strategy,
            "model": self.model,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved": [r.to_dict() for r in self.retrieved],
        }
