from unittest.mock import AsyncMock, patch

import pytest

from .config import GraphitiConfig, is_enabled_for_user

_ENV_VARS_TO_CLEAR = (
    "GRAPHITI_FALKORDB_HOST",
    "GRAPHITI_FALKORDB_PORT",
    "GRAPHITI_FALKORDB_PASSWORD",
    "CHAT_API_KEY",
    "CHAT_OPENAI_API_KEY",
    "OPEN_ROUTER_API_KEY",
    "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)


def test_graphiti_config_reads_backend_env_defaults() -> None:
    cfg = GraphitiConfig()

    assert cfg.falkordb_host == "localhost"
    assert cfg.falkordb_port == 6380


class TestResolveLlmApiKey:
    def test_returns_configured_key_when_set(self) -> None:
        cfg = GraphitiConfig(llm_api_key="my-llm-key")
        assert cfg.resolve_llm_api_key() == "my-llm-key"

    def test_falls_back_to_chat_api_key_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CHAT_API_KEY", "autopilot-key")
        monkeypatch.setenv("OPEN_ROUTER_API_KEY", "platform-key")
        cfg = GraphitiConfig(llm_api_key="")
        assert cfg.resolve_llm_api_key() == "autopilot-key"

    def test_falls_back_to_open_router_when_no_chat_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPEN_ROUTER_API_KEY", "fallback-router-key")
        cfg = GraphitiConfig(llm_api_key="")
        assert cfg.resolve_llm_api_key() == "fallback-router-key"

    def test_returns_empty_when_no_fallback(self) -> None:
        cfg = GraphitiConfig(llm_api_key="")
        assert cfg.resolve_llm_api_key() == ""


class TestResolveLlmBaseUrl:
    def test_returns_configured_url_when_set(self) -> None:
        cfg = GraphitiConfig(llm_base_url="https://custom.api/v1")
        assert cfg.resolve_llm_base_url() == "https://custom.api/v1"

    def test_falls_back_to_openrouter_base_url(self) -> None:
        cfg = GraphitiConfig(llm_base_url="")
        result = cfg.resolve_llm_base_url()
        assert result == "https://openrouter.ai/api/v1"


class TestResolveEmbedderApiKey:
    def test_returns_configured_key_when_set(self) -> None:
        cfg = GraphitiConfig(embedder_api_key="my-embedder-key")
        assert cfg.resolve_embedder_api_key() == "my-embedder-key"

    def test_falls_back_to_chat_openai_api_key_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CHAT_OPENAI_API_KEY", "autopilot-openai-key")
        monkeypatch.setenv("OPENAI_API_KEY", "platform-openai-key")
        cfg = GraphitiConfig(embedder_api_key="")
        assert cfg.resolve_embedder_api_key() == "autopilot-openai-key"

    def test_falls_back_to_openai_when_no_chat_openai_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "fallback-openai-key")
        cfg = GraphitiConfig(embedder_api_key="")
        assert cfg.resolve_embedder_api_key() == "fallback-openai-key"

    def test_returns_empty_when_no_fallback(self) -> None:
        cfg = GraphitiConfig(embedder_api_key="")
        assert cfg.resolve_embedder_api_key() == ""


class TestResolveEmbedderBaseUrl:
    def test_returns_configured_url_when_set(self) -> None:
        cfg = GraphitiConfig(embedder_base_url="https://embed.custom/v1")
        assert cfg.resolve_embedder_base_url() == "https://embed.custom/v1"

    def test_returns_none_when_empty(self) -> None:
        cfg = GraphitiConfig(embedder_base_url="")
        assert cfg.resolve_embedder_base_url() is None


class TestIsEnabledForUser:
    @pytest.mark.asyncio
    async def test_none_user_returns_false(self) -> None:
        result = await is_enabled_for_user(None)
        assert result is False

    @pytest.mark.asyncio
    async def test_empty_user_returns_false(self) -> None:
        result = await is_enabled_for_user("")
        assert result is False

    @pytest.mark.asyncio
    async def test_delegates_to_feature_flag(self) -> None:
        with patch(
            "backend.util.feature_flag.is_feature_enabled",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await is_enabled_for_user("some-user-id")
        assert result is True
