from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32


class ChunkingConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 80


class RetrievalConfig(BaseModel):
    top_k: int = 5
    dense_weight: float = 0.55
    bm25_weight: float = 0.45
    rrf_k: int = 60
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20


class GenerationConfig(BaseModel):
    temperature: float = 0.1
    max_context_chars: int = 6000
    max_tokens: int = 512


class PathsConfig(BaseModel):
    chroma_dir: str = "artifacts/chroma"
    collection_name: str = "docs"
    eval_dataset: str = "data/eval/qa_pairs.json"
    eval_report_dir: str = "artifacts/evals"


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class AppConfig(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # groq | huggingface | openai | mock  (empty = auto-detect from available keys)
    llm_provider: str = "groq"

    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-20b"

    hf_token: str = ""
    hf_base_url: str = "https://router.huggingface.co/v1"
    hf_model: str = "Qwen/Qwen2.5-7B-Instruct:fastest"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    rag_mock_llm: bool = False
    rag_config: str = "config.yaml"

    @field_validator("groq_api_key", "hf_token", "openai_api_key", mode="before")
    @classmethod
    def _strip_secret(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        cleaned = value.strip().strip("'").strip('"').strip()
        return cleaned


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_config(config_path: str | None = None) -> AppConfig:
    settings = get_settings()
    path = Path(config_path or settings.rag_config)
    return AppConfig.model_validate(_load_yaml(path))


def project_root() -> Path:
    return Path.cwd()
