"""Tests for environment-based LLM provider configuration."""

import pytest


def test_deepseek_profile_uses_direct_api_defaults() -> None:
    from backend.util.llm.config import resolve_chat_config

    config = resolve_chat_config(
        {
            "CHAT_PROVIDER": "deepseek",
            "CHAT_API_KEY": "secret",
        }
    )

    assert config.provider == "deepseek"
    assert config.base_url == "https://api.deepseek.com"
    assert config.model == "deepseek-v4-flash"
    assert config.fast_standard_model == "deepseek-v4-flash"
    assert config.fast_advanced_model == "deepseek-v4-pro"
    assert config.fallback_model == "deepseek-v4-pro"
    assert config.api_key_source == "CHAT_API_KEY"
    assert "api_key" not in config.model_dump()


def test_deepseek_explicit_model_stays_bare() -> None:
    from backend.util.llm.config import resolve_chat_config

    config = resolve_chat_config(
        {
            "CHAT_PROVIDER": "deepseek",
            "CHAT_API_KEY": "secret",
            "CHAT_MODEL": "deepseek-v4-pro",
        }
    )

    assert config.model == "deepseek-v4-pro"
    assert not config.model.startswith("deepseek/")


@pytest.mark.parametrize(
    ("base_url", "provider"),
    [
        ("https://api.deepseek.com", "deepseek"),
        ("https://openrouter.ai/api/v1", "openrouter"),
        ("https://api.openai.com/v1", "openai"),
    ],
)
def test_provider_is_inferred_from_known_base_url(base_url: str, provider: str) -> None:
    from backend.util.llm.config import resolve_chat_config

    config = resolve_chat_config(
        {
            "CHAT_BASE_URL": base_url,
            "CHAT_API_KEY": "secret",
        }
    )

    assert config.provider == provider


def test_openrouter_model_identifier_is_not_rewritten() -> None:
    from backend.util.llm.config import resolve_chat_config

    config = resolve_chat_config(
        {
            "CHAT_PROVIDER": "openrouter",
            "CHAT_API_KEY": "secret",
            "CHAT_MODEL": "deepseek/deepseek-v4-flash",
        }
    )

    assert config.base_url == "https://openrouter.ai/api/v1"
    assert config.model == "deepseek/deepseek-v4-flash"


def test_request_policy_defaults_and_overrides() -> None:
    from backend.util.llm.config import resolve_chat_config

    defaults = resolve_chat_config(
        {"CHAT_PROVIDER": "deepseek", "CHAT_API_KEY": "secret"}
    )
    overridden = resolve_chat_config(
        {
            "CHAT_PROVIDER": "deepseek",
            "CHAT_API_KEY": "secret",
            "CHAT_REQUEST_TIMEOUT_S": "7.5",
            "CHAT_MAX_RETRIES": "2",
        }
    )

    assert defaults.request_timeout_s == 20
    assert defaults.max_retries == 1
    assert overridden.request_timeout_s == 7.5
    assert overridden.max_retries == 2


def test_legacy_local_mode_is_preserved() -> None:
    from backend.util.llm.config import resolve_chat_config

    config = resolve_chat_config(
        {
            "CHAT_USE_LOCAL": "true",
            "CHAT_BASE_URL": "http://localhost:11434/v1",
            "CHAT_API_KEY": "ollama",
            "CHAT_MODEL": "qwen3:8b",
        }
    )

    assert config.provider == "local"
    assert config.base_url == "http://localhost:11434/v1"
    assert config.model == "qwen3:8b"
