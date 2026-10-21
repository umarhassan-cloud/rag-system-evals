from __future__ import annotations

from rag.config import AppConfig, Settings, get_settings
from rag.llm import build_llm_endpoint
from rag.models import AnswerResult, Citation, RetrievedChunk


SYSTEM_PROMPT = """You are a careful document Q&A assistant.
Answer ONLY using the provided context passages.
Cite sources inline using bracket markers like [1], [2] that match the passage numbers.
If the context is insufficient, say you don't know based on the provided documents.
Keep answers concise and factual."""


def build_context(retrieved: list[RetrievedChunk], max_chars: int) -> str:
    parts: list[str] = []
    used = 0
    for item in retrieved:
        block = f"[{item.rank}] Source: {item.chunk.source}\n{item.chunk.text}"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def citations_from_retrieved(retrieved: list[RetrievedChunk]) -> list[Citation]:
    citations: list[Citation] = []
    for item in retrieved:
        excerpt = item.chunk.text
        if len(excerpt) > 220:
            excerpt = excerpt[:217].rstrip() + "..."
        citations.append(
            Citation(
                index=item.rank,
                source=item.chunk.source,
                chunk_id=item.chunk.chunk_id,
                doc_id=item.chunk.doc_id,
                excerpt=excerpt,
                score=item.score,
            )
        )
    return citations


def _mock_answer(question: str, retrieved: list[RetrievedChunk]) -> str:
    if not retrieved:
        return "I don't know based on the provided documents."
    top = retrieved[0]
    return (
        f"Based on the retrieved documents, the most relevant evidence is in "
        f"{top.chunk.source} [{top.rank}]. "
        f"Key excerpt: {top.chunk.text[:280].strip()}"
        + ("..." if len(top.chunk.text) > 280 else "")
        + f" (Question: {question})"
    )


class Generator:
    def __init__(self, config: AppConfig, settings: Settings | None = None):
        self.config = config
        self.settings = settings or get_settings()
        self.endpoint = build_llm_endpoint(self.settings)

    @property
    def model_name(self) -> str:
        if self.endpoint.provider == "mock":
            return "mock-llm"
        return f"{self.endpoint.provider}:{self.endpoint.model}"

    def answer(self, question: str, retrieved: list[RetrievedChunk], strategy: str) -> AnswerResult:
        context = build_context(retrieved, self.config.generation.max_context_chars)
        if self.endpoint.client is None:
            text = _mock_answer(question, retrieved)
        else:
            text = self._llm_answer(question, context)
        return AnswerResult(
            question=question,
            answer=text,
            strategy=strategy,
            citations=citations_from_retrieved(retrieved),
            retrieved=retrieved,
            model=self.model_name,
        )

    def _llm_answer(self, question: str, context: str) -> str:
        assert self.endpoint.client is not None
        try:
            response = self.endpoint.client.chat.completions.create(
                model=self.endpoint.model,
                temperature=self.config.generation.temperature,
                max_tokens=self.config.generation.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Context passages:\n{context}\n\n"
                            f"Question: {question}\n\n"
                            "Answer with inline citations."
                        ),
                    },
                ],
            )
        except Exception as exc:
            provider = self.endpoint.provider
            hint = {
                "groq": "Check GROQ_API_KEY at https://console.groq.com/keys (no quotes in .env).",
                "huggingface": (
                    "Check HF_TOKEN at https://huggingface.co/settings/tokens "
                    "(create a token with 'Make calls to Inference Providers' permission, no quotes)."
                ),
                "openai": "Check OPENAI_API_KEY in .env (no quotes).",
            }.get(provider, "Check your LLM API key in .env.")
            raise RuntimeError(
                f"{provider} LLM request failed: {exc}\n{hint}"
            ) from exc
        return (response.choices[0].message.content or "").strip()
