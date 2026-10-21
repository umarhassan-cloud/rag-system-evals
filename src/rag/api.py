from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.pipeline import RAGPipeline
from rag.retrieval import STRATEGIES

app = FastAPI(title="RAG Q&A", version="0.1.0")
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    strategy: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = "hybrid"
    top_k: int | None = Field(default=None, ge=1, le=20)


class IngestRequest(BaseModel):
    path: str
    reset: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/strategies")
def list_strategies() -> dict[str, list[str]]:
    return {"strategies": list(STRATEGIES)}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if req.strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.strategy}")
    result = get_pipeline().ask(req.question, strategy=req.strategy, top_k=req.top_k)
    return result.to_dict()


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict[str, int]:
    try:
        return get_pipeline().ingest([req.path], reset=req.reset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
