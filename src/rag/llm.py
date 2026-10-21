from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from rag.config import Settings


@dataclass(frozen=True)
class LLMEndpoint:
    provider: str
    model: str
    client: OpenAI | None


PROVIDER_DEFAULTS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-20b",
    },
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        # OpenAI-compatible router; ":fastest" picks a free/fast provider when available.
        "model": "Qwen/Qwen2.5-7B-Instruct:fastest",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
}


def resolve_provider_name(settings: Settings) -> str:
    explicit = (settings.llm_provider or "").strip().lower()
    if explicit in {"groq", "huggingface", "hf", "openai", "mock"}:
        return "huggingface" if explicit == "hf" else explicit

    # Auto-pick a free provider when possible.
    if settings.groq_api_key:
        return "groq"
    if settings.hf_token:
        return "huggingface"
    if settings.openai_api_key:
        return "openai"
    return "mock"


def build_llm_endpoint(settings: Settings) -> LLMEndpoint:
    if settings.rag_mock_llm:
        return LLMEndpoint(provider="mock", model="mock-llm", client=None)

    provider = resolve_provider_name(settings)
    if provider == "mock":
        return LLMEndpoint(provider="mock", model="mock-llm", client=None)

    if provider == "groq":
        api_key = settings.groq_api_key
        base_url = settings.groq_base_url or PROVIDER_DEFAULTS["groq"]["base_url"]
        model = settings.groq_model or PROVIDER_DEFAULTS["groq"]["model"]
        if not api_key:
            return LLMEndpoint(provider="mock", model="mock-llm", client=None)
        return LLMEndpoint(
            provider="groq",
            model=model,
            client=OpenAI(api_key=api_key, base_url=base_url),
        )

    if provider == "huggingface":
        api_key = settings.hf_token
        base_url = settings.hf_base_url or PROVIDER_DEFAULTS["huggingface"]["base_url"]
        model = settings.hf_model or PROVIDER_DEFAULTS["huggingface"]["model"]
        if not api_key:
            return LLMEndpoint(provider="mock", model="mock-llm", client=None)
        return LLMEndpoint(
            provider="huggingface",
            model=model,
            client=OpenAI(api_key=api_key, base_url=base_url),
        )

    # openai
    api_key = settings.openai_api_key
    base_url = settings.openai_base_url or PROVIDER_DEFAULTS["openai"]["base_url"]
    model = settings.openai_model or PROVIDER_DEFAULTS["openai"]["model"]
    if not api_key:
        return LLMEndpoint(provider="mock", model="mock-llm", client=None)
    return LLMEndpoint(
        provider="openai",
        model=model,
        client=OpenAI(api_key=api_key, base_url=base_url),
    )
