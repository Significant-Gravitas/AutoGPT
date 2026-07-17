from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.config import ChatConfig

from .config import GraphitiConfig, is_enabled_for_user

_ENV_VARS_TO_CLEAR = (
    "GRAPHITI_FALKORDB_HOST",
    "GRAPHITI_FALKORDB_PORT",
    "GRAPHITI_FALKORDB_PASSWORD",
    "CHAT_API_KEY",
    "CHAT_OPENAI_API_KEY",
    "OPEN_ROUTER_API_KEY",
    "OPENAI_API_KEY",
    "CHAT_USE_LOCAL",
    "CHAT_USE_OPENROUTER",
    "CHAT_USE_CLAUDE_CODE_SUBSCRIPTION",
    "CHAT_BASE_URL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)


def _patch_chat_cfg(monkeypatch: pytest.MonkeyPatch, cfg: ChatConfig) -> None:
    """Replace the ``copilot.sdk.env.config`` singleton for one test.

    GraphitiConfig resolvers lazy-import ``chat_cfg`` inside
    ``_chat_local_fallback``; patching the module attribute is the
    cleanest way to pin a specific transport per test.
    """
    from backend.copilot.sdk import env

    monkeypatch.setattr(env, "config", cfg)


def _local_chat_cfg(
    api_key: str = "ollama-placeholder",
    base_url: str = "http://localhost:11434/v1",
) -> ChatConfig:
    return ChatConfig(use_local=True, api_key=api_key, base_url=base_url)


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

    def test_inherits_chat_api_key_under_local_transport(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under ``CHAT_USE_LOCAL=true`` with no graphiti override and
        no cloud key, the resolver falls through to the chat-config
        placeholder so the Ollama backend gets a non-empty bearer."""
        _patch_chat_cfg(monkeypatch, _local_chat_cfg(api_key="ollama-x"))
        cfg = GraphitiConfig(llm_api_key="")
        assert cfg.resolve_llm_api_key() == "ollama-x"

    def test_cloud_key_still_wins_over_local_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``CHAT_API_KEY`` (the AutoPilot-dedicated cloud key)
        beats the local fallback — covers the mixed-mode case where
        an operator switched the chat layer to local but left a
        cloud key around."""
        monkeypatch.setenv("CHAT_API_KEY", "autopilot-cloud-key")
        _patch_chat_cfg(monkeypatch, _local_chat_cfg(api_key="ollama-x"))
        cfg = GraphitiConfig(llm_api_key="")
        assert cfg.resolve_llm_api_key() == "autopilot-cloud-key"


class TestResolveLlmBaseUrl:
    def test_returns_configured_url_when_set(self) -> None:
        cfg = GraphitiConfig(llm_base_url="https://custom.api/v1")
        assert cfg.resolve_llm_base_url() == "https://custom.api/v1"

    def test_falls_back_to_openrouter_base_url(self) -> None:
        cfg = GraphitiConfig(llm_base_url="")
        result = cfg.resolve_llm_base_url()
        assert result == "https://openrouter.ai/api/v1"

    def test_inherits_chat_base_url_under_local_transport(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under local transport, graphiti's entity extraction must
        hit the same Ollama (or vLLM, …) endpoint as the chat layer
        — not the OpenRouter default. Otherwise the per-turn graphiti
        ingest 401s against a placeholder key."""
        _patch_chat_cfg(
            monkeypatch,
            _local_chat_cfg(base_url="http://ollama.lan:11434/v1"),
        )
        cfg = GraphitiConfig(llm_base_url="")
        assert cfg.resolve_llm_base_url() == "http://ollama.lan:11434/v1"


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

    def test_inherits_chat_api_key_under_local_transport(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_chat_cfg(monkeypatch, _local_chat_cfg(api_key="ollama-y"))
        cfg = GraphitiConfig(embedder_api_key="")
        assert cfg.resolve_embedder_api_key() == "ollama-y"


class TestResolveEmbedderBaseUrl:
    def test_returns_configured_url_when_set(self) -> None:
        cfg = GraphitiConfig(embedder_base_url="https://embed.custom/v1")
        assert cfg.resolve_embedder_base_url() == "https://embed.custom/v1"

    def test_returns_none_when_empty(self) -> None:
        cfg = GraphitiConfig(embedder_base_url="")
        assert cfg.resolve_embedder_base_url() is None

    def test_inherits_chat_base_url_under_local_transport(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Local transport directs embeddings to the same self-hosted
        backend. Operators still need to pull an embedding-capable
        model into Ollama (e.g. ``nomic-embed-text``) — covered in
        docs/platform/copilot-local-llm.md."""
        _patch_chat_cfg(
            monkeypatch,
            _local_chat_cfg(base_url="http://ollama.lan:11434/v1"),
        )
        cfg = GraphitiConfig(embedder_base_url="")
        assert cfg.resolve_embedder_base_url() == "http://ollama.lan:11434/v1"


class TestApplyLocalGraphitiModels:
    """``_apply_local_graphiti_models`` rewires cloud model defaults
    to local-friendly slugs under ``CHAT_USE_LOCAL=true`` so a fresh
    install of AutoGPT + Ollama works end-to-end without setting a
    ``GRAPHITI_*_MODEL`` env var per slot. Mirrors dev's
    ``_apply_local_aux_models`` pattern on the chat side."""

    def test_rewrites_cloud_defaults_under_local(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_chat_cfg(monkeypatch, _local_chat_cfg())
        cfg = GraphitiConfig()
        # llm_model: gpt-4.1-mini → Qwen3.5-4B Unsloth GGUF
        assert cfg.llm_model == "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M"
        # reranker reuses the same model (simpler prompts, avoids a
        # second Ollama pull).
        assert cfg.reranker_model == "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M"
        # embedder: text-embedding-3-small → nomic-embed-text
        assert cfg.embedder_model == "nomic-embed-text"

    def test_operator_override_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Operators with custom Ollama tags (``qwen3:8b``,
        ``hf.co/...``) keep their override — the validator only
        rewrites *literal* cloud defaults."""
        _patch_chat_cfg(monkeypatch, _local_chat_cfg())
        monkeypatch.setenv("GRAPHITI_LLM_MODEL", "qwen3:8b")
        monkeypatch.setenv("GRAPHITI_EMBEDDER_MODEL", "all-minilm")
        cfg = GraphitiConfig()
        assert cfg.llm_model == "qwen3:8b"
        assert cfg.embedder_model == "all-minilm"

    def test_cloud_defaults_preserved_under_cloud_transport(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenRouter / direct-Anthropic transports leave the cloud
        defaults alone — graphiti's ``gpt-4.1-mini`` stays put for
        anyone who isn't on a self-hosted backend."""
        _patch_chat_cfg(
            monkeypatch,
            ChatConfig(
                use_openrouter=True,
                api_key="or-key",
                base_url="https://openrouter.ai/api/v1",
            ),
        )
        cfg = GraphitiConfig()
        assert cfg.llm_model == "gpt-4.1-mini"
        assert cfg.reranker_model == "gpt-4.1-nano"
        assert cfg.embedder_model == "text-embedding-3-small"


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
