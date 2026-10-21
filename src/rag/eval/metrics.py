from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag.models import RetrievedChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


@dataclass
class EvalExample:
    id: str
    question: str
    relevant_doc_ids: list[str]
    relevant_contains: list[str]
    reference_answer: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvalExample":
        return cls(
            id=str(raw["id"]),
            question=str(raw["question"]),
            relevant_doc_ids=[str(x) for x in raw.get("relevant_doc_ids", [])],
            relevant_contains=[str(x) for x in raw.get("relevant_contains", [])],
            reference_answer=str(raw.get("reference_answer", "")),
        )


def load_eval_dataset(path: Path) -> list[EvalExample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Eval dataset must be a JSON list")
    return [EvalExample.from_dict(item) for item in data]


def chunk_is_relevant(chunk_text: str, source: str, example: EvalExample) -> bool:
    source_l = source.lower()
    if any(doc.lower() in source_l or source_l in doc.lower() for doc in example.relevant_doc_ids):
        # Still require at least one content cue when provided, to avoid overly loose hits.
        if not example.relevant_contains:
            return True
    text_l = chunk_text.lower()
    return any(needle.lower() in text_l for needle in example.relevant_contains)


def hit_at_k(retrieved: list[RetrievedChunk], example: EvalExample, k: int) -> float:
    for item in retrieved[:k]:
        if chunk_is_relevant(item.chunk.text, item.chunk.source, example):
            return 1.0
    return 0.0


def reciprocal_rank(retrieved: list[RetrievedChunk], example: EvalExample) -> float:
    for item in retrieved:
        if chunk_is_relevant(item.chunk.text, item.chunk.source, example):
            return 1.0 / float(item.rank)
    return 0.0


def context_recall(retrieved: list[RetrievedChunk], example: EvalExample) -> float:
    if not example.relevant_contains:
        return hit_at_k(retrieved, example, k=len(retrieved) or 1)
    hits = 0
    for needle in example.relevant_contains:
        if any(needle.lower() in item.chunk.text.lower() for item in retrieved):
            hits += 1
    return hits / len(example.relevant_contains)


def context_precision(retrieved: list[RetrievedChunk], example: EvalExample) -> float:
    if not retrieved:
        return 0.0
    relevant = sum(
        1 for item in retrieved if chunk_is_relevant(item.chunk.text, item.chunk.source, example)
    )
    return relevant / len(retrieved)


def answer_token_f1(prediction: str, reference: str) -> float:
    if not reference.strip():
        return 0.0
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred or not ref:
        return 0.0
    overlap = len(pred & ref)
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def faithfulness_proxy(answer: str, retrieved: list[RetrievedChunk]) -> float:
    """Lightweight grounding check: fraction of answer content tokens found in context."""
    answer_tokens = tokenize(answer)
    # Drop ultra-common stop-ish tokens for a more meaningful overlap.
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "based", "documents"}
    answer_tokens = {t for t in answer_tokens if t not in stop and len(t) > 2}
    if not answer_tokens:
        return 1.0 if retrieved else 0.0
    context = " ".join(item.chunk.text for item in retrieved).lower()
    grounded = sum(1 for t in answer_tokens if t in context)
    return grounded / len(answer_tokens)


@dataclass
class StrategyMetrics:
    strategy: str
    n: int
    hit_at_5: float
    mrr: float
    context_recall: float
    context_precision: float
    answer_f1: float
    faithfulness: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate(strategy: str, rows: list[dict[str, float]]) -> StrategyMetrics:
    def avg(key: str) -> float:
        if not rows:
            return 0.0
        return sum(r[key] for r in rows) / len(rows)

    return StrategyMetrics(
        strategy=strategy,
        n=len(rows),
        hit_at_5=avg("hit_at_5"),
        mrr=avg("mrr"),
        context_recall=avg("context_recall"),
        context_precision=avg("context_precision"),
        answer_f1=avg("answer_f1"),
        faithfulness=avg("faithfulness"),
    )
