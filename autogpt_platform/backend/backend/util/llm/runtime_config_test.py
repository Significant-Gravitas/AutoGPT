from datetime import UTC, datetime

import pytest
from pydantic import SecretStr

from backend.data.llm_provider_settings import PersistedLlmProviderSettings
from backend.util.llm.config import resolve_chat_config
from backend.util.llm.runtime_config import (
    LlmRuntimeOverrides,
    merge_effective_llm_config,
)


def _persisted(**updates) -> PersistedLlmProviderSettings:
    now = datetime.now(UTC)
    values = {
        "id": "default",
        "enabled": True,
        "provider": "deepseek",
        "use_local": False,
        "base_url": "https://api.deepseek.com",
        "encrypted_api_key": "ciphertext",
        "model": "deepseek-v4-flash",
        "title_model": "deepseek-v4-flash",
        "fast_standard_model": "deepseek-v4-flash",
        "fast_advanced_model": "deepseek-v4-pro",
        "thinking_standard_model": "deepseek-v4-flash",
        "thinking_advanced_model": "deepseek-v4-pro",
        "claude_agent_fallback_model": "deepseek-v4-pro",
        "request_timeout_s": 20,
        "max_retries": 1,
        "local_request_timeout_s": 20,
        "store_embedding_model": "text-embedding-3-small",
        "created_at": now,
        "updated_at": now,
    }
    values.update(updates)
    return PersistedLlmProviderSettings(**values)


def test_env_only_fallback_preserves_deepseek() -> None:
    env = {
        "CHAT_PROVIDER": "deepseek",
        "CHAT_API_KEY": "env-secret",
        "CHAT_MODEL": "deepseek-v4-flash",
        "CHAT_LOCAL_REQUEST_TIMEOUT_S": "20",
    }
    config = merge_effective_llm_config(
        resolve_chat_config(env),
        environment=env,
    )

    assert config.provider == "deepseek"
    assert config.source == "env"
    assert config.request_timeout_s == 20
    assert config.local_request_timeout_s == 20


def test_persisted_settings_override_environment_and_mask_key() -> None:
    env = {
        "CHAT_PROVIDER": "openrouter",
        "CHAT_BASE_URL": "https://openrouter.ai/api/v1",
        "CHAT_API_KEY": "env-secret",
        "CHAT_MODEL": "openai/gpt-4o-mini",
    }
    config = merge_effective_llm_config(
        resolve_chat_config(env),
        environment=env,
        persisted=_persisted(),
        persisted_api_key="database-secret",
    )

    assert config.provider == "deepseek"
    assert config.model == "deepseek-v4-flash"
    assert config.source == "db"
    assert config.source_by_field["provider"] == "db"
    assert config.api_key_configured is True
    assert config.api_key_masked == "********"
    assert "api_key" not in config.model_dump()
    assert "database-secret" not in config.model_dump_json()


def test_openrouter_model_identifier_is_not_normalized_to_direct_deepseek() -> None:
    env = {
        "CHAT_PROVIDER": "openrouter",
        "CHAT_API_KEY": "secret",
        "CHAT_MODEL": "deepseek/deepseek-v4-flash",
    }
    config = merge_effective_llm_config(
        resolve_chat_config(env),
        environment=env,
    )

    assert config.provider == "openrouter"
    assert config.model == "deepseek/deepseek-v4-flash"


def test_direct_deepseek_rejects_openrouter_model_identifier() -> None:
    env = {"CHAT_PROVIDER": "openrouter", "CHAT_API_KEY": "secret"}

    with pytest.raises(ValueError, match="bare model IDs"):
        merge_effective_llm_config(
            resolve_chat_config(env),
            environment=env,
            persisted=_persisted(model="deepseek/deepseek-v4-flash"),
        )


def test_local_timeout_below_60_only_rejected_for_local_provider() -> None:
    cloud_env = {
        "CHAT_PROVIDER": "deepseek",
        "CHAT_API_KEY": "secret",
        "CHAT_LOCAL_REQUEST_TIMEOUT_S": "20",
    }
    assert resolve_chat_config(cloud_env).local_request_timeout_s == 20

    local_env = {
        "CHAT_PROVIDER": "local",
        "CHAT_BASE_URL": "http://localhost:11434/v1",
        "CHAT_API_KEY": "ollama",
        "CHAT_MODEL": "qwen3:8b",
        "CHAT_LOCAL_REQUEST_TIMEOUT_S": "20",
    }
    with pytest.raises(ValueError, match="at least 60"):
        resolve_chat_config(local_env)


def test_unsaved_runtime_override_wins_without_changing_openrouter_ids() -> None:
    env = {
        "CHAT_PROVIDER": "openrouter",
        "CHAT_API_KEY": "openrouter-secret",
        "CHAT_MODEL": "deepseek/deepseek-v4-flash",
    }
    config = merge_effective_llm_config(
        resolve_chat_config(env),
        environment=env,
        overrides=LlmRuntimeOverrides(
            provider="deepseek",
            base_url="https://api.deepseek.com",
            api_key=SecretStr("unsaved-secret"),
            model="deepseek-v4-pro",
            title_model="deepseek-v4-flash",
            fast_standard_model="deepseek-v4-flash",
            fast_advanced_model="deepseek-v4-pro",
            thinking_standard_model="deepseek-v4-flash",
            thinking_advanced_model="deepseek-v4-pro",
            claude_agent_fallback_model="deepseek-v4-pro",
        ),
    )

    assert config.source == "override"
    assert config.provider == "deepseek"
    assert config.model == "deepseek-v4-pro"
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == "unsaved-secret"


def test_empty_runtime_override_preserves_environment_source() -> None:
    env = {
        "CHAT_PROVIDER": "deepseek",
        "CHAT_API_KEY": "secret",
        "CHAT_MODEL": "deepseek-v4-flash",
    }

    config = merge_effective_llm_config(
        resolve_chat_config(env),
        environment=env,
        overrides=LlmRuntimeOverrides(),
    )

    assert config.source == "env"
    assert "override" not in config.source_by_field.values()
