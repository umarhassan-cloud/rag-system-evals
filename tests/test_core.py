from __future__ import annotations

from rag.ingest import chunk_text, normalize_whitespace
from rag.eval.metrics import (
    EvalExample,
    answer_token_f1,
    context_precision,
    hit_at_k,
    reciprocal_rank,
)
from rag.models import DocumentChunk, RetrievedChunk
from rag.retrieval.bm25 import BM25Index, reciprocal_rank_fusion


def test_normalize_and_chunk():
    text = normalize_whitespace("Hello   world.\n\n\nNext  paragraph.")
    chunks = chunk_text(text, doc_id="d1", source="s.md", chunk_size=40, chunk_overlap=5)
    assert chunks
    assert all(c.doc_id == "d1" for c in chunks)
    assert chunks[0].source == "s.md"


def test_bm25_and_rrf():
    chunks = [
        DocumentChunk("c1", "d1", "a.md", "annual refund money-back guarantee"),
        DocumentChunk("c2", "d2", "b.md", "api rate limits growth plan"),
        DocumentChunk("c3", "d3", "c.md", "encryption aes-256 at rest"),
    ]
    index = BM25Index(chunks)
    hits = index.search("refund money-back", top_k=2)
    assert hits[0].chunk.chunk_id == "c1"

    dense = [
        RetrievedChunk(chunks[1], 0.9, 1, "dense"),
        RetrievedChunk(chunks[0], 0.8, 2, "dense"),
    ]
    fused = reciprocal_rank_fusion([hits, dense], top_k=2)
    assert len(fused) == 2
    assert {x.chunk.chunk_id for x in fused} <= {"c1", "c2"}


def test_retrieval_metrics():
    example = EvalExample(
        id="1",
        question="refund?",
        relevant_doc_ids=["billing_policy.md"],
        relevant_contains=["14-day money-back guarantee"],
        reference_answer="Annual plans have a 14-day money-back guarantee.",
    )
    good = DocumentChunk("1", "d", "billing_policy.md", "Annual plans include a 14-day money-back guarantee.")
    bad = DocumentChunk("2", "d", "other.md", "Totally unrelated text about cats.")
    retrieved = [
        RetrievedChunk(good, 1.0, 1, "dense"),
        RetrievedChunk(bad, 0.1, 2, "dense"),
    ]
    assert hit_at_k(retrieved, example, 1) == 1.0
    assert reciprocal_rank(retrieved, example) == 1.0
    assert context_precision(retrieved, example) == 0.5
    assert answer_token_f1(
        "Annual plans have a 14-day money-back guarantee.",
        example.reference_answer,
    ) > 0.5
