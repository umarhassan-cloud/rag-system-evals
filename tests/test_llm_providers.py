from rag.llm import build_llm_endpoint, resolve_provider_name


def _settings(**overrides):
    # Bypass dotenv/env so unit tests are deterministic.
    from rag.config import Settings

    values = {
        "llm_provider": "groq",
        "groq_api_key": "",
        "groq_base_url": "https://api.groq.com/openai/v1",
        "groq_model": "openai/gpt-oss-20b",
        "hf_token": "",
        "hf_base_url": "https://router.huggingface.co/v1",
        "hf_model": "Qwen/Qwen2.5-7B-Instruct:fastest",
        "openai_api_key": "",
        "openai_base_url": "https://api.openai.com/v1",
        "openai_model": "gpt-4o-mini",
        "rag_mock_llm": False,
        "rag_config": "config.yaml",
    }
    values.update(overrides)
    return Settings.model_construct(**values)


def test_resolve_prefers_explicit_provider():
    settings = _settings(llm_provider="huggingface", hf_token="hf_x", groq_api_key="g_x")
    assert resolve_provider_name(settings) == "huggingface"


def test_build_groq_endpoint():
    settings = _settings(llm_provider="groq", groq_api_key="gsk_test", groq_model="openai/gpt-oss-20b")
    endpoint = build_llm_endpoint(settings)
    assert endpoint.provider == "groq"
    assert endpoint.model == "openai/gpt-oss-20b"
    assert endpoint.client is not None


def test_missing_key_falls_back_to_mock():
    settings = _settings(llm_provider="groq", groq_api_key="")
    endpoint = build_llm_endpoint(settings)
    assert endpoint.provider == "mock"
    assert endpoint.client is None
