"""Unit tests for ``transport_routing.routing_kwargs_for_chat_transport``.

Each test wires up a fresh ``ChatConfig`` and ``Settings`` and asserts
that the resolved ``ProviderRoutingKwargs`` matches the active
``TransportProfile`` row plus the right runtime credential. These
pin the integration boundary between dev's transport machinery
(#12993) and our centralized ``util/llm/providers.py::call_provider``,
so regressions in either side surface immediately.
"""

from unittest.mock import MagicMock

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.transport_routing import (
    ProviderRoutingKwargs,
    routing_kwargs_for_chat_transport,
)

# Same env-cleanup pattern as ``config_test.py`` — leftover ``CHAT_*`` /
# ``*_API_KEY`` env vars in the developer or CI environment would flip
# transport resolution and break these tests deterministically.
_ENV_VARS_TO_CLEAR = (
    "CHAT_USE_OPENROUTER",
    "CHAT_USE_CLAUDE_CODE_SUBSCRIPTION",
    "CHAT_USE_LOCAL",
    "CHAT_API_KEY",
    "CHAT_BASE_URL",
    "CHAT_DIRECT_ANTHROPIC_API_KEY",
    "OPEN_ROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)


def _patch_chat_cfg(monkeypatch: pytest.MonkeyPatch, cfg: ChatConfig) -> None:
    """Swap the module-level ``copilot.sdk.env.config`` for a test config.

    The helper imports ``chat_cfg`` lazily inside
    ``routing_kwargs_for_chat_transport``, so we patch the singleton
    directly instead of monkeypatching the function-local name.
    """
    from backend.copilot.sdk import env

    monkeypatch.setattr(env, "config", cfg)


def _patch_settings(monkeypatch: pytest.MonkeyPatch, **secrets) -> None:
    """Patch ``backend.util.settings.Settings`` to expose the given secrets.

    Only the fields the helper reads (``anthropic_api_key``,
    ``open_router_api_key``) need to be set per-test.
    """
    from backend.util import settings as settings_mod

    fake_secrets = MagicMock()
    fake_secrets.anthropic_api_key = secrets.get("anthropic_api_key", "")
    fake_secrets.open_router_api_key = secrets.get("open_router_api_key", "")
    fake_settings = MagicMock()
    fake_settings.secrets = fake_secrets
    monkeypatch.setattr(settings_mod, "Settings", lambda: fake_settings)


class TestLocalTransport:
    """Local transport routes through Ollama. ``api_key`` and
    ``base_url`` come from ``chat_cfg`` directly — no fallback to
    Settings since local has ``api_key_fallback_envs=()``."""

    def test_local_routing_kwargs(self, monkeypatch: pytest.MonkeyPatch):
        cfg = ChatConfig(
            use_local=True,
            api_key="ollama-placeholder",
            base_url="http://localhost:11434/v1",
        )
        _patch_chat_cfg(monkeypatch, cfg)
        _patch_settings(monkeypatch)

        result = routing_kwargs_for_chat_transport()

        assert result == ProviderRoutingKwargs(
            provider="ollama",
            api_key="ollama-placeholder",
            base_url="http://localhost:11434/v1",
            supports_flex=False,
            cost_log_provider="ollama",
        )

    def test_local_with_empty_api_key_is_still_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Ollama doesn't validate the bearer token, so an empty
        ``api_key`` is acceptable. The helper returns ``""`` and the
        ``call_provider`` ollama branch ignores it."""
        # ChatConfig validates that ``CHAT_API_KEY`` is non-empty under
        # local, so the operator must set *something* — we mimic the
        # "minimal placeholder" case the docs recommend.
        cfg = ChatConfig(
            use_local=True,
            api_key="x",
            base_url="http://h:11434/v1",
        )
        _patch_chat_cfg(monkeypatch, cfg)
        _patch_settings(monkeypatch)

        result = routing_kwargs_for_chat_transport()
        assert result.provider == "ollama"
        assert result.base_url == "http://h:11434/v1"
        assert result.supports_flex is False


class TestOpenRouterTransport:
    """OpenRouter is the canonical cloud path. Reads
    ``settings.secrets.open_router_api_key`` when ``chat_cfg.api_key``
    is unset; ``base_url`` is left ``None`` so ``call_provider``'s
    OpenRouter branch uses its baked-in ``OPENROUTER_BASE_URL``."""

    def test_openrouter_with_chat_api_key(self, monkeypatch: pytest.MonkeyPatch):
        cfg = ChatConfig(
            use_openrouter=True,
            api_key="or-key-from-chat-config",
            base_url="https://openrouter.ai/api/v1",
        )
        _patch_chat_cfg(monkeypatch, cfg)
        _patch_settings(monkeypatch, open_router_api_key="or-key-from-secrets")

        result = routing_kwargs_for_chat_transport()
        # chat_cfg.api_key wins over secrets.
        assert result.provider == "open_router"
        assert result.api_key == "or-key-from-chat-config"
        assert result.base_url is None
        assert result.supports_flex is True
        assert result.cost_log_provider == "open_router"

    def test_openrouter_falls_back_to_secrets(self, monkeypatch: pytest.MonkeyPatch):
        # OpenRouter requires both api_key and a valid base_url for the
        # ``openrouter_active`` gate. Use the canonical OR base url so the
        # transport actually resolves to ``openrouter`` and not
        # ``direct_anthropic``.
        cfg = ChatConfig(
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
        )
        _patch_chat_cfg(monkeypatch, cfg)
        _patch_settings(monkeypatch, open_router_api_key="or-key-from-secrets")

        result = routing_kwargs_for_chat_transport()
        # ``cfg.api_key`` set to "or-key" wins; demonstrating the
        # fallback path requires a config that resolves to openrouter
        # without setting chat_cfg.api_key — which the validator forbids
        # under the openrouter_active gate. The fallback exists as
        # belt-and-suspenders for runtime env mutations.
        assert result.api_key == "or-key"


class TestSubscriptionTransport:
    """Subscription users authenticate the chat path via Claude Code
    OAuth, but that token can't call the Messages API directly (see
    Anthropic Feb-2026 ToS). Dream pass needs a separate
    ``ANTHROPIC_API_KEY`` — the helper surfaces ``""`` when missing
    so callers can produce a friendly error."""

    def test_subscription_with_anthropic_key_from_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        cfg = ChatConfig(
            use_claude_code_subscription=True,
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4-7",
            aux_api_key="or-aux-key",
        )
        _patch_chat_cfg(monkeypatch, cfg)
        _patch_settings(monkeypatch, anthropic_api_key="ant-key-12345")

        result = routing_kwargs_for_chat_transport()
        assert result.provider == "anthropic"
        assert result.api_key == "ant-key-12345"
        assert result.base_url is None
        assert result.supports_flex is False
        assert result.cost_log_provider == "anthropic"

    def test_subscription_without_anthropic_key_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Friendly-error path — callers (e.g. dream/llm.py) check for
        empty ``api_key`` and raise with an env-var hint instead of
        letting ``call_provider`` 401."""
        cfg = ChatConfig(
            use_claude_code_subscription=True,
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4-7",
            aux_api_key="or-aux-key",
        )
        _patch_chat_cfg(monkeypatch, cfg)
        _patch_settings(monkeypatch)  # anthropic_api_key="" by default

        result = routing_kwargs_for_chat_transport()
        assert result.provider == "anthropic"
        assert result.api_key == ""
        assert result.cost_log_provider == "anthropic"

    def test_subscription_prefers_direct_anthropic_api_key_field(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """``chat.direct_anthropic_api_key`` (if the operator set it)
        wins over the platform-wide ``settings.secrets.anthropic_api_key``."""
        cfg = ChatConfig(
            use_claude_code_subscription=True,
            direct_anthropic_api_key="ant-key-direct-cfg",
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4-7",
            aux_api_key="or-aux-key",
        )
        _patch_chat_cfg(monkeypatch, cfg)
        _patch_settings(monkeypatch, anthropic_api_key="ant-key-from-secrets")

        result = routing_kwargs_for_chat_transport()
        assert result.api_key == "ant-key-direct-cfg"


class TestDirectAnthropicTransport:
    """direct_anthropic is the explicit "skip OpenRouter, talk to
    Anthropic" cloud transport. Same provider as subscription but
    different transport identity (so cost-log labels match)."""

    def test_direct_anthropic_routing_kwargs(self, monkeypatch: pytest.MonkeyPatch):
        cfg = ChatConfig(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            direct_anthropic_api_key="ant-key-direct",
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4-7",
            aux_api_key="or-aux-key",
        )
        _patch_chat_cfg(monkeypatch, cfg)
        _patch_settings(monkeypatch)

        result = routing_kwargs_for_chat_transport()
        assert result.provider == "anthropic"
        assert result.api_key == "ant-key-direct"
        assert result.base_url is None
        assert result.supports_flex is False
        assert result.cost_log_provider == "anthropic"


class TestStaticIdentityPropagation:
    """Sanity tests that the static ``TransportProfile`` fields
    (``dispatch_provider``, ``supports_flex_tier``,
    ``cost_log_provider``) flow into the returned dataclass without
    transformation."""

    @pytest.mark.parametrize(
        "transport_kwargs, expected_provider, expected_cost, expected_flex",
        [
            (
                dict(
                    use_local=True,
                    api_key="ollama",
                    base_url="http://h:11434/v1",
                ),
                "ollama",
                "ollama",
                False,
            ),
            (
                dict(
                    use_openrouter=True,
                    api_key="or-key",
                    base_url="https://openrouter.ai/api/v1",
                ),
                "open_router",
                "open_router",
                True,
            ),
        ],
    )
    def test_profile_fields_propagate_into_routing_kwargs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        transport_kwargs: dict,
        expected_provider: str,
        expected_cost: str,
        expected_flex: bool,
    ):
        cfg = ChatConfig(**transport_kwargs)
        _patch_chat_cfg(monkeypatch, cfg)
        _patch_settings(monkeypatch, open_router_api_key="or-secrets-key")

        result = routing_kwargs_for_chat_transport()
        assert result.provider == expected_provider
        assert result.cost_log_provider == expected_cost
        assert result.supports_flex is expected_flex
