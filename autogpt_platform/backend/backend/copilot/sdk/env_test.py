"""The CLI environment each route profile launches with (``route_launch_env``)."""

from unittest.mock import patch

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.route_profile import (
    RouteProfile,
    codex_profile,
    platform_profile,
    user_key_profile,
)
from backend.copilot.sdk.context_window import CodexEngineWindow
from backend.copilot.sdk.env import (
    GatewayEndpoint,
    describe_sdk_context,
    route_launch_env,
)
from backend.copilot.sdk.route_credentials import CLAUDE_OAUTH_BLANK

_SONNET = "anthropic/claude-sonnet-4-6"
_GATEWAY = GatewayEndpoint(base_url="http://127.0.0.1:43123", auth_token="cap-token")
_HARDENING = {
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_ENABLE_TASKS": "0",
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
}


def _config(**overrides) -> ChatConfig:
    return ChatConfig(**overrides)


def _openrouter(model: str = _SONNET, **overrides) -> RouteProfile:
    cfg = _config(
        platform_route="openrouter",
        openrouter_api_key="sk-or-test-key",
        **overrides,
    )
    return platform_profile(cfg, model, "env")


def _direct(model: str = "claude-sonnet-4-6", **overrides) -> RouteProfile:
    cfg = _config(
        platform_route="direct_anthropic", anthropic_api_key="sk-ant-test", **overrides
    )
    return platform_profile(cfg, model, "env")


def _subscription(model: str = "claude-sonnet-4-6") -> RouteProfile:
    return platform_profile(_config(platform_route="subscription"), model, "env")


def _codex(
    model: str = "gpt-5.6-terra", engine: CodexEngineWindow | None = None
) -> RouteProfile:
    return codex_profile(
        model, effort="high", routing_source="catalog", codex_engine=engine
    )


class TestSubscription:
    @patch("backend.copilot.sdk.env.validate_subscription")
    def test_blanks_every_key_and_url(self, mock_validate):
        result = route_launch_env(_subscription())

        assert result["ANTHROPIC_API_KEY"] == ""
        assert result["ANTHROPIC_AUTH_TOKEN"] == ""
        assert result["ANTHROPIC_BASE_URL"] == ""
        # The route fills its 200K window, not the platform's cost trigger.
        assert result["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "90"
        assert result["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "200000"
        # Anthropic first-party: the CLI keeps its attribution header.
        assert "CLAUDE_CODE_ATTRIBUTION_HEADER" not in result
        mock_validate.assert_called_once()

    @patch(
        "backend.copilot.sdk.env.validate_subscription",
        side_effect=RuntimeError("CLI not found"),
    )
    def test_propagates_validation_error(self, _mock_validate):
        with pytest.raises(RuntimeError, match="CLI not found"):
            route_launch_env(_subscription())

    @patch("backend.copilot.sdk.env.validate_subscription")
    def test_inherited_base_url_is_blanked(self, _mock_validate):
        """The CLI sends the OAuth bearer to any base URL, so a parent
        ``ANTHROPIC_BASE_URL`` must be overridden, not inherited."""
        with patch.dict(
            "backend.copilot.sdk.env.os.environ",
            {"ANTHROPIC_BASE_URL": "https://remote-gateway.example.com"},
            clear=True,
        ):
            result = route_launch_env(_subscription())
        assert result["ANTHROPIC_BASE_URL"] == ""

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://remote-gateway.example.com",
            "https://api.anthropic.com",
            "http://gw.localhost:8080",
            "http://localtest.me:8080",
            "http://127.0.0.1.nip.io:8080",
            "https://127.0.0.1:8443",
        ],
    )
    def test_refuses_non_loopback_base_url(self, base_url):
        with (
            patch("backend.copilot.sdk.env.validate_subscription"),
            patch(
                "backend.copilot.sdk.env.subscription_env",
                return_value={"ANTHROPIC_BASE_URL": base_url},
            ),
        ):
            with pytest.raises(ValueError, match="loopback HTTP URL"):
                route_launch_env(_subscription())

    @pytest.mark.parametrize(
        "base_url", ["http://127.0.0.1:8099", "http://localhost:8099", "http://[::1]:1"]
    )
    def test_allows_loopback_base_url(self, base_url):
        with (
            patch("backend.copilot.sdk.env.validate_subscription"),
            patch(
                "backend.copilot.sdk.env.subscription_env",
                return_value={"ANTHROPIC_BASE_URL": base_url},
            ),
        ):
            result = route_launch_env(_subscription())
        assert result["ANTHROPIC_BASE_URL"] == base_url

    def test_absent_base_url_is_checked_against_the_parent_env(self):
        with (
            patch("backend.copilot.sdk.env.validate_subscription"),
            patch("backend.copilot.sdk.env.subscription_env", return_value={}),
            patch.dict(
                "backend.copilot.sdk.env.os.environ",
                {"ANTHROPIC_BASE_URL": "https://remote-gateway.example.com"},
                clear=True,
            ),
        ):
            with pytest.raises(ValueError, match="loopback HTTP URL"):
                route_launch_env(_subscription())


class TestDirectAnthropic:
    def test_route_key_against_anthropics_api(self):
        result = route_launch_env(_direct())

        assert result["ANTHROPIC_API_KEY"] == "sk-ant-test"
        assert result["ANTHROPIC_AUTH_TOKEN"] == ""
        assert result["ANTHROPIC_BASE_URL"] == ""
        for name, value in CLAUDE_OAUTH_BLANK.items():
            assert result[name] == value
        assert "CLAUDE_CODE_ATTRIBUTION_HEADER" not in result
        assert result["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "1000000"
        assert "CLAUDE_CODE_DISABLE_1M_CONTEXT" not in result

    def test_configured_url_is_used_without_v1(self):
        result = route_launch_env(
            _direct(anthropic_base_url="https://anthropic.example.com/v1/")
        )
        assert result["ANTHROPIC_BASE_URL"] == "https://anthropic.example.com"

    def test_overrides_inherited_credentials_and_base_url(self):
        parent = {
            "ANTHROPIC_AUTH_TOKEN": "inherited-token",
            "ANTHROPIC_BASE_URL": "https://inherited.example.com",
            "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-inherited",
        }
        with patch.dict("backend.copilot.sdk.env.os.environ", parent, clear=True):
            result = route_launch_env(_direct())
        merged = {**parent, **result}
        assert merged["ANTHROPIC_AUTH_TOKEN"] == ""
        assert merged["ANTHROPIC_BASE_URL"] == ""
        assert merged["CLAUDE_CODE_OAUTH_TOKEN"] == ""

    def test_refuses_without_the_routes_key(self):
        profile = platform_profile(
            _config(platform_route="direct_anthropic"), "claude-sonnet-4-6", "env"
        )
        with pytest.raises(RuntimeError, match="CHAT_ANTHROPIC_API_KEY"):
            route_launch_env(profile)


class TestOpenRouter:
    def test_route_key_as_bearer_without_the_attribution_header(self):
        result = route_launch_env(_openrouter())

        assert result["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
        assert result["ANTHROPIC_AUTH_TOKEN"] == "sk-or-test-key"
        assert result["ANTHROPIC_API_KEY"] == ""
        for name, value in CLAUDE_OAUTH_BLANK.items():
            assert result[name] == value
        # Per-session values in the CLI's billing block would make every
        # session's system prefix different.
        assert result["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
        assert result["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "200000"
        assert result["CLAUDE_CODE_DISABLE_1M_CONTEXT"] == "1"
        assert result["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "50"

    def test_inherited_oauth_token_never_reaches_the_cli(self):
        parent = {
            "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-inherited",
            "CLAUDE_CODE_REFRESH_TOKEN": "inherited-refresh",
        }
        with patch.dict("backend.copilot.sdk.env.os.environ", parent, clear=True):
            result = route_launch_env(_openrouter())
        merged = {**parent, **result}
        assert merged["ANTHROPIC_AUTH_TOKEN"] == "sk-or-test-key"
        assert merged["CLAUDE_CODE_OAUTH_TOKEN"] == ""
        assert merged["CLAUDE_CODE_REFRESH_TOKEN"] == ""

    @pytest.mark.parametrize(
        "url",
        [
            "https://openrouter.ai/api/v1",
            "https://openrouter.ai/api/v1/",
            "https://openrouter.ai/api",
        ],
    )
    def test_base_url_without_v1(self, url):
        result = route_launch_env(_openrouter(openrouter_base_url=url))
        assert result["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"

    def test_refuses_without_the_routes_key(self):
        profile = platform_profile(_config(platform_route="openrouter"), _SONNET, "env")
        with pytest.raises(RuntimeError, match="CHAT_OPENROUTER_API_KEY"):
            route_launch_env(profile)

    def test_session_and_user_headers(self):
        result = route_launch_env(_openrouter(), session_id="sess-1", user_id="user-2")
        headers = result["ANTHROPIC_CUSTOM_HEADERS"].split("\n")
        assert headers == [
            "x-session-id: sess-1",
            "x-user-id: user-2",
            "Accept-Encoding: identity",
        ]

    def test_header_values_are_sanitised_and_truncated(self):
        result = route_launch_env(
            _openrouter(), session_id="a\nb\x7fc" + "x" * 200, user_id="ü-id"
        )
        headers = result["ANTHROPIC_CUSTOM_HEADERS"].split("\n")
        assert headers[0] == "x-session-id: " + ("abc" + "x" * 200)[:128]
        assert headers[1] == "x-user-id: -id"

    def test_other_routes_send_no_trace_headers(self):
        result = route_launch_env(_direct(), session_id="sess-1", user_id="user-2")
        assert result["ANTHROPIC_CUSTOM_HEADERS"] == "Accept-Encoding: identity"


class TestEveryRoute:
    @pytest.mark.parametrize("route", ["openrouter", "direct", "subscription", "codex"])
    def test_hardening_tmpdir_and_timeout(self, route):
        profiles = {
            "openrouter": _openrouter,
            "direct": _direct,
            "subscription": _subscription,
            "codex": _codex,
        }
        with patch("backend.copilot.sdk.env.validate_subscription"):
            result = route_launch_env(
                profiles[route](),
                sdk_cwd="/tmp/copilot-abc",
                gateway=_GATEWAY if route == "codex" else None,
            )
        for name, value in _HARDENING.items():
            assert result[name] == value
        assert result["CLAUDE_CODE_TMPDIR"] == "/tmp/copilot-abc"
        assert result["API_TIMEOUT_MS"] == "600000"
        assert result["ANTHROPIC_CUSTOM_HEADERS"].endswith("Accept-Encoding: identity")

    def test_no_tmpdir_without_a_working_directory(self):
        assert "CLAUDE_CODE_TMPDIR" not in route_launch_env(_openrouter(), sdk_cwd="")

    def test_a_gateway_route_without_its_gateway_is_refused(self):
        with pytest.raises(RuntimeError, match="behind a gateway"):
            route_launch_env(_codex())


class TestTrigger:
    @pytest.mark.parametrize(
        "model", ["anthropic/claude-sonnet-5", "anthropic/claude-sonnet-5-5"]
    )
    def test_sonnet_5_is_scaled_for_its_tokenizer(self, model):
        assert route_launch_env(_openrouter(model))[
            "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"
        ] == ("65")

    def test_direct_sonnet_5_is_scaled_too(self):
        result = route_launch_env(_direct("claude-sonnet-5-5"))
        assert result["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "65"

    @pytest.mark.parametrize("model", ["moonshotai/kimi-k2.6", "moonshotai/kimi-k3"])
    def test_moonshot_keeps_the_clis_trigger(self, model):
        assert "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE" not in route_launch_env(
            _openrouter(model)
        )

    def test_moonshot_is_held_to_its_own_window(self):
        from backend.copilot.moonshot import moonshot_context_window

        model = "moonshotai/kimi-k2.6"
        window = min(200_000, moonshot_context_window(model) or 200_000)
        result = route_launch_env(_openrouter(model))
        assert result["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == str(window)


class TestCodex:
    def test_gateway_credentials_and_the_engine_default(self):
        result = route_launch_env(_codex(), gateway=_GATEWAY)

        assert result["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:43123"
        assert result["ANTHROPIC_AUTH_TOKEN"] == "cap-token"
        assert result["ANTHROPIC_API_KEY"] == ""
        assert result["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "272000"
        # Lifted with the pin, or the CLI's 200K assumption clamps it away.
        assert result["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "272000"
        assert result["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "90"
        assert result["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
        assert "CLAUDE_CODE_DISABLE_1M_CONTEXT" not in result

    def test_follows_the_accounts_advertised_window_and_limit(self):
        engine = CodexEngineWindow(
            context_window=400_000, auto_compact_token_limit=300_000
        )
        result = route_launch_env(_codex(engine=engine), gateway=_GATEWAY)
        assert result["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "400000"
        assert result["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "400000"
        assert result["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "75"

    def test_a_moonshot_shaped_slug_is_still_codex(self):
        result = route_launch_env(_codex("moonshotai/kimi-k2.6"), gateway=_GATEWAY)
        assert result["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "90"
        assert result["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "272000"

    def test_preserves_existing_no_proxy_hosts(self):
        with patch.dict(
            "backend.copilot.sdk.env.os.environ",
            {"NO_PROXY": "corp.internal,localhost"},
            clear=True,
        ):
            result = route_launch_env(_codex(), gateway=_GATEWAY)
        assert result["NO_PROXY"] == result["no_proxy"]
        entries = result["NO_PROXY"].split(",")
        for host in (
            "corp.internal",
            "localhost",
            "127.0.0.1",
            "::1",
        ):
            assert host in entries

    def test_refuses_a_non_loopback_gateway(self):
        gateway = GatewayEndpoint(base_url="https://gw.example.com", auth_token="t")
        with pytest.raises(ValueError, match="loopback HTTP URL"):
            route_launch_env(_codex(), gateway=gateway)

    def test_refuses_an_empty_capability(self):
        gateway = GatewayEndpoint(base_url="http://127.0.0.1:1", auth_token="")
        with pytest.raises(ValueError, match="must not be empty"):
            route_launch_env(_codex(), gateway=gateway)

    def test_non_gateway_routes_leave_the_model_table_alone(self):
        assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in route_launch_env(_openrouter())


class TestOrchestratorLaunchMatchesChat:
    """The orchestrator block launches through the same builder: its env for a
    route is the chat's for that route, minus what attributes the chat's
    session (its headers, and on OpenRouter its Broadcast body)."""

    @pytest.mark.parametrize(
        ("provider", "chat_profile", "key"),
        [
            ("open_router", lambda: _openrouter(), "sk-or-test-key"),
            ("anthropic", lambda: _direct(_SONNET), "sk-ant-test"),
        ],
    )
    def test_same_env_as_the_chat_minus_its_session_attribution(
        self, provider, chat_profile, key
    ):
        chat = route_launch_env(
            chat_profile(), sdk_cwd="/tmp/w", session_id="sess-1", user_id="user-2"
        )
        block = route_launch_env(
            user_key_profile(provider, key, _SONNET), sdk_cwd="/tmp/w"
        )
        chat_body = chat.pop("CLAUDE_CODE_EXTRA_BODY", None)
        assert (chat_body is not None) == (provider == "open_router")
        assert "CLAUDE_CODE_EXTRA_BODY" not in block
        chat_headers = chat.pop("ANTHROPIC_CUSTOM_HEADERS").split("\n")
        block_headers = block.pop("ANTHROPIC_CUSTOM_HEADERS").split("\n")
        assert chat == block
        session = [
            h for h in chat_headers if h.startswith(("x-session-id", "x-user-id"))
        ]
        assert [h for h in chat_headers if h not in session] == block_headers

    def test_a_users_openrouter_key_never_goes_to_the_chats_own_url(self):
        """A self-hosted chat may point CHAT_OPENROUTER_BASE_URL anywhere; the
        user's own key still goes only to OpenRouter."""
        block = route_launch_env(user_key_profile("open_router", "sk-user", _SONNET))
        assert block["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
        assert block["ANTHROPIC_AUTH_TOKEN"] == "sk-user"

    def test_a_users_anthropic_key_goes_only_to_anthropic(self):
        block = route_launch_env(user_key_profile("anthropic", "sk-ant-user", _SONNET))
        assert block["ANTHROPIC_BASE_URL"] == ""
        assert block["ANTHROPIC_API_KEY"] == "sk-ant-user"


class TestDescribeSdkContext:
    def test_codex_route_summary(self):
        env = route_launch_env(_codex(), gateway=_GATEWAY)
        line = describe_sdk_context(
            route="codex", model="gpt-5.6-terra", sdk_env=env, window_source="account"
        )
        assert line == (
            "route=codex model=gpt-5.6-terra window=272000 window_source=account "
            "trigger_pct=90 disable_1m_context=false"
        )

    def test_platform_route_with_kill_switch_and_default_trigger(self):
        env = route_launch_env(_openrouter("moonshotai/kimi-k2.6"))
        line = describe_sdk_context(route="openrouter", model="m", sdk_env=env)
        assert "trigger_pct=<cli-default>" in line
        assert "disable_1m_context=true" in line

    def test_never_leaks_secrets(self):
        env = route_launch_env(_openrouter(), session_id="s")
        line = describe_sdk_context(route="openrouter", model="m", sdk_env=env)
        assert "sk-or-test-key" not in line
