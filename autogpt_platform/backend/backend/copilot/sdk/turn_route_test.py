"""Which route serves an SDK turn: its profile, the model the platform route
serves for a tier, and the gateway a route launches behind."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.copilot.config import ChatConfig
from backend.copilot.model_router import ResolvedModel
from backend.copilot.route_profile import codex_profile, platform_profile
from backend.copilot.route_table import RoutingSource
from backend.copilot.sdk.model_gateway.route import (
    ApiKeyCredential,
    CodexLeaseCredential,
    NoCredential,
)
from backend.copilot.sdk.turn_route import (
    resolve_platform_model,
    resolve_turn_profile,
    start_route_gateway,
    turn_route_name,
)
from backend.integrations.codex.transport import CodexCredentialLease

_ROUTER = "backend.copilot.sdk.turn_route.resolve_model_route"


def _routed(model: str, source: RoutingSource = "ld"):
    return patch(_ROUTER, new=AsyncMock(return_value=ResolvedModel(model, source)))


class TestResolvePlatformModel:
    """The platform route's model for a tier: flag, catalog, then the tier,
    as the route writes it. A flag-served name the route cannot serve falls
    back to the tier's own model, never the other tier's."""

    @pytest.mark.asyncio
    async def test_openrouter_takes_any_vendor(self):
        cfg = ChatConfig(platform_route="openrouter", openrouter_api_key="k")
        with _routed("moonshotai/kimi-k2.6"):
            resolved = await resolve_platform_model(cfg, "standard", "s", "u")
        assert resolved == ("moonshotai/kimi-k2.6", "ld")

    @pytest.mark.asyncio
    async def test_direct_route_refuses_another_vendor_and_falls_back(self):
        """A Kimi flag value on the direct route would fail the turn; the
        tier's own model serves instead, stamped as the env layer's."""
        cfg = ChatConfig(
            platform_route="direct_anthropic",
            standard_model="anthropic/claude-sonnet-4-6",
        )
        with _routed("moonshotai/kimi-k2.6"):
            resolved = await resolve_platform_model(cfg, "standard", "s", "u")
        assert resolved == ("claude-sonnet-4-6", "env")

    @pytest.mark.asyncio
    async def test_advanced_fallback_is_the_advanced_model(self):
        """Falling back to the standard tier would silently downgrade the
        user's chosen tier."""
        cfg = ChatConfig(
            platform_route="direct_anthropic",
            standard_model="anthropic/claude-sonnet-4-6",
            advanced_model="anthropic/claude-opus-4.7",
        )
        with _routed("moonshotai/kimi-k2.6"):
            resolved = await resolve_platform_model(cfg, "advanced", "s", "u")
        assert resolved == ("claude-opus-4-7", "env")

    @pytest.mark.asyncio
    async def test_a_tier_the_route_cannot_serve_either_raises(self):
        cfg = ChatConfig(
            platform_route="direct_anthropic", standard_model="openai/gpt-5.5"
        )
        with _routed("moonshotai/kimi-k2.6"), pytest.raises(ValueError):
            await resolve_platform_model(cfg, "standard", "s", "u")

    @pytest.mark.asyncio
    async def test_subscription_sends_the_native_id(self):
        """The subscription CLI rejects ``anthropic/claude-opus-4.7``."""
        cfg = ChatConfig(platform_route="subscription")
        with _routed("anthropic/claude-opus-4.7"):
            resolved = await resolve_platform_model(cfg, "advanced", "s", "u")
        assert resolved == ("claude-opus-4-7", "ld")

    @pytest.mark.asyncio
    async def test_subscription_names_its_standard_model_too(self):
        """The CLI is told which model to run on every tier, never left to
        pick its own default."""
        cfg = ChatConfig(platform_route="subscription")
        with patch(
            "backend.copilot.model_router.get_feature_flag_value",
            new=AsyncMock(return_value=None),
        ):
            resolved = await resolve_platform_model(cfg, "standard", "s", "u")
        assert resolved == ("claude-sonnet-5-5", "env")

    @pytest.mark.asyncio
    async def test_subscription_refuses_another_vendor_and_falls_back(self):
        cfg = ChatConfig(
            platform_route="subscription", standard_model="anthropic/claude-sonnet-4-6"
        )
        with _routed("moonshotai/kimi-k2.6"):
            resolved = await resolve_platform_model(cfg, "standard", "s", "u")
        assert resolved == ("claude-sonnet-4-6", "env")


class TestResolveTurnProfile:
    @pytest.mark.asyncio
    async def test_no_lease_runs_on_the_platform_route(self):
        cfg = ChatConfig(platform_route="openrouter", openrouter_api_key="k")
        with _routed("anthropic/claude-sonnet-4-6", "env"):
            profile = await resolve_turn_profile(
                cfg, model=None, session_id="s", user_id="u", credential_lease=None
            )
        assert (profile.name, profile.model, profile.routing_source) == (
            "openrouter",
            "anthropic/claude-sonnet-4-6",
            "env",
        )

    @pytest.mark.asyncio
    async def test_a_disabled_route_has_no_platform_turn(self):
        cfg = ChatConfig(platform_route="disabled")
        with (
            _routed("anthropic/claude-sonnet-4-6", "env"),
            pytest.raises(RuntimeError, match="CHAT_PLATFORM_ROUTE=disabled"),
        ):
            await resolve_turn_profile(
                cfg, model=None, session_id="s", user_id="u", credential_lease=None
            )

    @pytest.mark.asyncio
    async def test_only_a_codex_lease_is_taken(self):
        cfg = ChatConfig(platform_route="openrouter", openrouter_api_key="k")
        with pytest.raises(TypeError):
            await resolve_turn_profile(
                cfg,
                model=None,
                session_id="s",
                user_id="u",
                credential_lease=MagicMock(),
            )


class TestStartRouteGateway:
    """The seam that starts the shared model gateway: Codex on the account's
    lease, the local route on the operator's server through its adapter, and
    no gateway for the routes that speak the CLI's own protocol."""

    @pytest.mark.asyncio
    async def test_a_route_without_an_adapter_needs_no_gateway(self):
        cfg = ChatConfig(platform_route="openrouter", openrouter_api_key="k")
        profile = platform_profile(cfg, "anthropic/claude-sonnet-4-6", "env")
        assert await start_route_gateway(profile, None) is None

    @pytest.mark.asyncio
    async def test_codex_starts_its_gateway_on_the_lease(self):
        profile = codex_profile(
            "gpt-5.6-terra", effort="medium", routing_source="env", codex_engine=None
        )
        lease = MagicMock(spec=CodexCredentialLease)
        lease.credentials.id = "cred-1"
        gateway = MagicMock()
        gateway.start = AsyncMock()
        with patch(
            "backend.copilot.sdk.turn_route.ModelGateway", return_value=gateway
        ) as gateway_class:
            started = await start_route_gateway(profile, lease)
        assert started is gateway
        gateway.start.assert_awaited_once()
        [route] = gateway_class.call_args.args
        assert route.adapter == "codex_responses"
        assert (route.model, route.effort, route.base_url) == (
            "gpt-5.6-terra",
            "medium",
            None,
        )
        assert route.credential == CodexLeaseCredential(lease=lease)
        assert route.context_window == profile.window
        assert route.timeout_seconds == profile.row.api_timeout_ms / 1000

    @pytest.mark.asyncio
    async def test_codex_without_its_lease_is_refused(self):
        profile = codex_profile(
            "gpt-5.6-terra", effort=None, routing_source="env", codex_engine=None
        )
        with pytest.raises(RuntimeError, match="credential lease"):
            await start_route_gateway(profile, None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter, base_url, key",
        [
            ("openai_chat", "http://127.0.0.1:8080/v1", "sk-local"),
            ("anthropic_passthrough", "http://127.0.0.1:8080", None),
        ],
    )
    async def test_the_local_route_starts_its_gateway_on_the_operators_server(
        self, adapter, base_url, key
    ):
        cfg = ChatConfig(
            platform_route="local",
            local_base_url=base_url,
            local_adapter=adapter,
            local_api_key=key,
            local_context_tokens=131_072,
            standard_model="qwen3.6-35b",
            advanced_model="qwen3.6-35b",
        )
        profile = platform_profile(cfg, "qwen3.6-35b", "env")
        gateway = MagicMock()
        gateway.start = AsyncMock()
        with patch(
            "backend.copilot.sdk.turn_route.ModelGateway", return_value=gateway
        ) as gateway_class:
            started = await start_route_gateway(profile, None)
        assert started is gateway
        [route] = gateway_class.call_args.args
        assert (route.adapter, route.model, route.base_url) == (
            adapter,
            "qwen3.6-35b",
            base_url,
        )
        assert route.credential == (
            ApiKeyCredential(api_key=SecretStr(key)) if key else NoCredential()
        )
        assert route.context_window == 131_072
        assert route.timeout_seconds == 1800


def test_turn_route_name():
    cfg = ChatConfig(platform_route="openrouter", openrouter_api_key="k")
    profile = platform_profile(cfg, "anthropic/claude-sonnet-4-6", "env")
    assert turn_route_name(profile, None, cfg) == "openrouter"
    # A turn that ended before its profile was built.
    assert turn_route_name(None, MagicMock(), cfg) == "codex"
    assert turn_route_name(None, None, cfg) == "openrouter"
