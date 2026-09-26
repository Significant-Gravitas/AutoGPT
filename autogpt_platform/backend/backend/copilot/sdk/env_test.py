"""Tests for build_sdk_env() — the SDK subprocess environment builder."""

from unittest.mock import patch

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.sdk.context_window import CodexEngineWindow

# ---------------------------------------------------------------------------
# Helpers — build a ChatConfig with explicit field values so tests don't
# depend on real environment variables.
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> ChatConfig:
    """Create a ChatConfig with safe defaults, applying *overrides*.

    SDK model fields are pinned to anthropic/* so the
    ``_validate_sdk_model_vendor_compatibility`` model_validator allows
    construction with ``use_openrouter=False`` (the default here).
    """
    defaults = {
        "use_claude_code_subscription": False,
        "use_openrouter": False,
        # Explicit: init kwargs beat both process env and the .env file,
        # so a local-flavored developer .env can't flip the transport.
        "use_local": False,
        "api_key": None,
        "base_url": None,
        # Fast tiers pinned like the thinking tiers: the direct-Anthropic
        # vendor validator rejects non-anthropic slugs, and a
        # local-flavored .env would otherwise leak llama slugs in here.
        "fast_standard_model": "anthropic/claude-sonnet-5",
        "fast_advanced_model": "anthropic/claude-opus-4-8",
        "thinking_standard_model": "anthropic/claude-sonnet-4-6",
        "thinking_advanced_model": "anthropic/claude-opus-4-7",
        # Pinned: both are settable from the environment, and a developer's
        # .env otherwise rewrites what build_sdk_env() is asked to emit.
        "claude_agent_autocompact_pct_override": 50,
        # Unset so the route's engine default applies; tests for the
        # explicit override pass a value via overrides.
        "claude_agent_context_window": None,
        # Aux key satisfies ``_validate_aux_client_for_direct_main`` —
        # these tests target SDK behavior, not the aux check.
        "aux_api_key": "or-aux-key",
    }
    defaults.update(overrides)
    return ChatConfig(**defaults)


# ---------------------------------------------------------------------------
# Mode 1 — Subscription auth
# ---------------------------------------------------------------------------


class TestBuildSdkEnvSubscription:
    """When ``use_claude_code_subscription`` is True, keys are blanked."""

    @patch("backend.copilot.sdk.env.validate_subscription")
    def test_returns_blanked_keys(self, mock_validate):
        """Subscription mode clears API_KEY, AUTH_TOKEN, and BASE_URL."""
        cfg = _make_config(use_claude_code_subscription=True)
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env()

        assert result["ANTHROPIC_API_KEY"] == ""
        assert result["ANTHROPIC_AUTH_TOKEN"] == ""
        assert result["ANTHROPIC_BASE_URL"] == ""
        assert result.get("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS") == "1"
        assert result.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE") == "50"
        mock_validate.assert_called_once()

    @patch(
        "backend.copilot.sdk.env.validate_subscription",
        side_effect=RuntimeError("CLI not found"),
    )
    def test_propagates_validation_error(self, mock_validate):
        """If validate_subscription fails, the error bubbles up."""
        cfg = _make_config(use_claude_code_subscription=True)
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            with pytest.raises(RuntimeError, match="CLI not found"):
                build_sdk_env()


# ---------------------------------------------------------------------------
# Mode 2 — Direct Anthropic (no OpenRouter)
# ---------------------------------------------------------------------------


class TestBuildSdkEnvDirectAnthropic:
    """When OpenRouter is inactive, no ANTHROPIC_* overrides (inherit parent env)."""

    def test_no_anthropic_key_overrides_when_openrouter_inactive(self):
        cfg = _make_config(use_openrouter=False)
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env()

        assert "ANTHROPIC_API_KEY" not in result
        assert "ANTHROPIC_AUTH_TOKEN" not in result
        assert "ANTHROPIC_BASE_URL" not in result
        assert result.get("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS") == "1"
        assert result.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE") == "50"

    def test_no_anthropic_key_overrides_when_openrouter_flag_true_but_no_key(self):
        """OpenRouter flag is True but no api_key => openrouter_active is False."""
        cfg = _make_config(use_openrouter=True, base_url="https://openrouter.ai/api/v1")
        # Force api_key to None after construction (field_validator may pick up env vars)
        object.__setattr__(cfg, "api_key", None)
        assert not cfg.openrouter_active
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env()

        assert "ANTHROPIC_API_KEY" not in result
        assert "ANTHROPIC_AUTH_TOKEN" not in result
        assert "ANTHROPIC_BASE_URL" not in result
        assert result.get("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS") == "1"
        assert result.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE") == "50"


# ---------------------------------------------------------------------------
# Mode 3 — OpenRouter proxy
# ---------------------------------------------------------------------------


class TestBuildSdkEnvOpenRouter:
    """When OpenRouter is active, return proxy env vars."""

    def _openrouter_config(self, **overrides):
        defaults = {
            "use_openrouter": True,
            "api_key": "sk-or-test-key",
            "base_url": "https://openrouter.ai/api/v1",
        }
        defaults.update(overrides)
        return _make_config(**defaults)

    def test_basic_openrouter_env(self):
        cfg = self._openrouter_config()
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env()

        assert result["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
        assert result["ANTHROPIC_AUTH_TOKEN"] == "sk-or-test-key"
        assert result["ANTHROPIC_API_KEY"] == ""
        # SDK 0.1.58: Accept-Encoding: identity is always injected
        assert "ANTHROPIC_CUSTOM_HEADERS" in result
        assert "Accept-Encoding: identity" in result["ANTHROPIC_CUSTOM_HEADERS"]
        # OpenRouter compat: env var must always be present
        assert result.get("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS") == "1"
        assert result.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE") == "50"

    def test_strips_trailing_v1(self):
        """The /v1 suffix is stripped from the base URL."""
        cfg = self._openrouter_config(base_url="https://openrouter.ai/api/v1")
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env()

        assert result["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
        assert result.get("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS") == "1"

    def test_strips_trailing_v1_and_slash(self):
        """Trailing slash before /v1 strip is handled."""
        cfg = self._openrouter_config(base_url="https://openrouter.ai/api/v1/")
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env()

        # rstrip("/") first, then remove /v1
        assert result["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
        assert result.get("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS") == "1"

    def test_no_v1_suffix_left_alone(self):
        """A base URL without /v1 is used as-is."""
        cfg = self._openrouter_config(base_url="https://custom-proxy.example.com")
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env()

        assert result["ANTHROPIC_BASE_URL"] == "https://custom-proxy.example.com"
        assert result.get("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS") == "1"

    def test_session_id_header(self):
        cfg = self._openrouter_config()
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(session_id="sess-123")

        assert "ANTHROPIC_CUSTOM_HEADERS" in result
        assert "x-session-id: sess-123" in result["ANTHROPIC_CUSTOM_HEADERS"]

    def test_user_id_header(self):
        cfg = self._openrouter_config()
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(user_id="user-456")

        assert "x-user-id: user-456" in result["ANTHROPIC_CUSTOM_HEADERS"]

    def test_both_headers(self):
        cfg = self._openrouter_config()
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(session_id="s1", user_id="u2")

        headers = result["ANTHROPIC_CUSTOM_HEADERS"]
        assert "x-session-id: s1" in headers
        assert "x-user-id: u2" in headers
        # They should be newline-separated
        assert "\n" in headers

    def test_header_sanitisation_strips_newlines(self):
        """Newlines/carriage-returns in header values are stripped."""
        cfg = self._openrouter_config()
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(session_id="bad\r\nvalue")

        header_val = result["ANTHROPIC_CUSTOM_HEADERS"]
        # The _safe helper removes \r and \n
        assert "\r" not in header_val.split(": ", 1)[1]
        assert "badvalue" in header_val

    def test_header_value_truncated_to_128_chars(self):
        """Header values are truncated to 128 characters."""
        cfg = self._openrouter_config()
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            long_id = "x" * 200
            result = build_sdk_env(session_id=long_id)

        # SDK 0.1.58 appends Accept-Encoding: identity on a separate line.
        # Parse the x-session-id line specifically and check its value length.
        headers = result["ANTHROPIC_CUSTOM_HEADERS"]
        session_line = next(
            line for line in headers.splitlines() if line.startswith("x-session-id: ")
        )
        value = session_line.split(": ", 1)[1]
        assert len(value) == 128

    @pytest.mark.parametrize(
        ("bad_input", "expected_ascii"),
        [
            ("user\x00id", "userid"),  # null byte
            ("user\x7fid", "userid"),  # DEL
            ("user\x80id", "userid"),  # first C1 control char
            ("user\x9fid", "userid"),  # last C1 control char
            ("user\U0001f600id", "userid"),  # emoji (non-ASCII Unicode)
            ("user\u202eid", "userid"),  # RTL override (security-relevant)
        ],
    )
    def test_header_sanitizer_strips_non_printable_ascii(
        self, bad_input: str, expected_ascii: str
    ):
        """_safe() strips everything outside printable ASCII 0x20–0x7e."""
        cfg = self._openrouter_config()
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(session_id=bad_input)

        value = result["ANTHROPIC_CUSTOM_HEADERS"].split(": ", 1)[1]
        assert expected_ascii in value
        for char in bad_input:
            if ord(char) < 0x20 or ord(char) > 0x7E:
                assert char not in value


# ---------------------------------------------------------------------------
# Mode priority
# ---------------------------------------------------------------------------


class TestBuildSdkEnvModePriority:
    """Subscription mode takes precedence over OpenRouter."""

    @patch("backend.copilot.sdk.env.validate_subscription")
    def test_subscription_overrides_openrouter(self, mock_validate):
        cfg = _make_config(
            use_claude_code_subscription=True,
            use_openrouter=True,
            api_key="sk-or-key",
            base_url="https://openrouter.ai/api/v1",
        )
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env()

        # Should get subscription result (blanked keys), not OpenRouter proxy
        assert result["ANTHROPIC_API_KEY"] == ""
        assert result["ANTHROPIC_AUTH_TOKEN"] == ""
        assert result["ANTHROPIC_BASE_URL"] == ""
        # SDK 0.1.58: Accept-Encoding: identity is always injected — no trace headers
        assert result.get("ANTHROPIC_CUSTOM_HEADERS") == "Accept-Encoding: identity"


# ---------------------------------------------------------------------------
# CLAUDE_CODE_TMPDIR integration
# ---------------------------------------------------------------------------


class TestClaudeCodeTmpdir:
    """Verify build_sdk_env() sets CLAUDE_CODE_TMPDIR from *sdk_cwd*."""

    def test_tmpdir_set_when_sdk_cwd_is_truthy(self):
        """CLAUDE_CODE_TMPDIR is set to sdk_cwd when sdk_cwd is truthy."""
        cfg = _make_config(use_openrouter=False)
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(sdk_cwd="/tmp/copilot-workspace")

        assert result["CLAUDE_CODE_TMPDIR"] == "/tmp/copilot-workspace"

    def test_tmpdir_not_set_when_sdk_cwd_is_none(self):
        """CLAUDE_CODE_TMPDIR is NOT in the env when sdk_cwd is None."""
        cfg = _make_config(use_openrouter=False)
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(sdk_cwd=None)

        assert "CLAUDE_CODE_TMPDIR" not in result

    def test_tmpdir_not_set_when_sdk_cwd_is_empty_string(self):
        """CLAUDE_CODE_TMPDIR is NOT in the env when sdk_cwd is empty string."""
        cfg = _make_config(use_openrouter=False)
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(sdk_cwd="")

        assert "CLAUDE_CODE_TMPDIR" not in result

    @patch("backend.copilot.sdk.env.validate_subscription")
    def test_tmpdir_set_in_subscription_mode(self, mock_validate):
        """CLAUDE_CODE_TMPDIR is set even in subscription mode."""
        cfg = _make_config(use_claude_code_subscription=True)
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(sdk_cwd="/tmp/sub-workspace")

        assert result["CLAUDE_CODE_TMPDIR"] == "/tmp/sub-workspace"
        assert result["ANTHROPIC_API_KEY"] == ""


# ---------------------------------------------------------------------------
# CLAUDE_AUTOCOMPACT_PCT_OVERRIDE — Moonshot gate
# ---------------------------------------------------------------------------


class TestAutocompactPctOverrideMoonshotGate:
    """Override is set for Anthropic / unknown models, skipped for Moonshot.

    Moonshot's OpenRouter endpoint silently drops cache writes
    (cache_create=0 in observed traces), so the 50% threshold's
    cache-cost rationale doesn't apply there.  Forcing aggressive
    compaction made the CLI auto-compact 3+ times per turn against
    Kimi's larger effective window — each compaction added a slow
    extra LLM round-trip.
    """

    @pytest.mark.parametrize(
        "model",
        [
            None,
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-opus-4-7",
            "claude-sonnet-4-6",
        ],
    )
    def test_override_set_for_non_moonshot(self, model):
        cfg = _make_config(use_openrouter=False)
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model=model)

        assert result.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE") == "50"

    @pytest.mark.parametrize(
        "model",
        [
            "moonshotai/kimi-k2.6",
            "moonshotai/kimi-k2.5",
            "moonshotai/kimi-k3.0",
        ],
    )
    def test_override_skipped_for_moonshot(self, model):
        cfg = _make_config(
            use_openrouter=True,
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
            thinking_standard_model=model,
        )
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model=model)

        assert "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE" not in result


class TestAutocompactPctOverrideConfigurable:
    """The override percentage is read from
    ``claude_agent_autocompact_pct_override`` so operators can tune it per
    deployment.  Setting to 0 omits the env var entirely (CLI uses its
    ~93% default), useful when the post-compact floor (system prompt +
    tool defs ≈ 65-110K) sits close to an aggressive trigger and
    cascading recompactions show up.
    """

    @pytest.mark.parametrize("pct", [25, 50, 70, 93])
    def test_config_value_propagates_to_env(self, pct):
        cfg = _make_config(
            use_openrouter=False, claude_agent_autocompact_pct_override=pct
        )
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model="anthropic/claude-sonnet-4-6")

        assert result.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE") == str(pct)

    def test_zero_omits_env_var(self):
        cfg = _make_config(
            use_openrouter=False, claude_agent_autocompact_pct_override=0
        )
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model="anthropic/claude-sonnet-4-6")

        assert "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE" not in result

    def test_moonshot_still_skipped_regardless_of_config(self):
        cfg = _make_config(
            use_openrouter=True,
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
            thinking_standard_model="moonshotai/kimi-k2.6",
            claude_agent_autocompact_pct_override=70,
        )
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model="moonshotai/kimi-k2.6")

        assert "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE" not in result

    def test_pct_override_rejects_out_of_range(self):
        """Pydantic bounds (ge=0, le=100) prevent invalid percentages so the
        env var never receives garbage."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _make_config(claude_agent_autocompact_pct_override=101)
        with pytest.raises(ValidationError):
            _make_config(claude_agent_autocompact_pct_override=-1)

    def test_override_set_when_model_is_none(self):
        """When build_sdk_env is called without a resolved model (e.g. very
        early init paths), default to setting the env var — Anthropic-default
        behaviour is the safe choice since most non-Moonshot routes benefit."""
        cfg = _make_config(use_openrouter=False)
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model=None)

        assert result.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE") == "50"


# ---------------------------------------------------------------------------
# Defensive guard — local transport must never reach the SDK env builder
# ---------------------------------------------------------------------------


class TestBuildSdkEnvLocalTransportGuard:
    """``use_local=True`` is incompatible with the SDK CLI's Anthropic
    wire protocol, so reaching ``build_sdk_env`` under that transport
    indicates an upstream routing bug (the request layer should have
    downgraded to baseline). The builder fails loudly rather than
    constructing a doomed subprocess env."""

    def test_local_transport_raises(self):
        cfg = _make_config(
            use_local=True,
            api_key="ollama",
            base_url="http://host.docker.internal:11434/v1",
        )
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            with pytest.raises(
                RuntimeError, match=r"transport 'local'.*doesn't support the SDK"
            ):
                build_sdk_env()

    def test_codex_override_runs_sdk_under_local_transport(self):
        cfg = _make_config(
            use_local=True,
            api_key="ollama",
            base_url="http://host.docker.internal:11434/v1",
        )
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(
                codex_gateway_url="http://127.0.0.1:43210/",
                codex_gateway_token="loopback-capability",
            )

        assert result["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:43210"
        assert result["ANTHROPIC_AUTH_TOKEN"] == "loopback-capability"
        assert result["ANTHROPIC_API_KEY"] == ""
        assert result["CLAUDE_CODE_OAUTH_TOKEN"] == ""
        assert result["CLAUDE_CODE_REFRESH_TOKEN"] == ""
        assert {"127.0.0.1", "localhost", "::1"} <= set(result["NO_PROXY"].split(","))
        assert result["no_proxy"] == result["NO_PROXY"]


class TestBuildSdkEnvCodexGateway:
    def test_codex_override_preserves_existing_no_proxy_hosts(self):
        cfg = _make_config(use_openrouter=False)
        with (
            patch("backend.copilot.sdk.env.config", cfg),
            patch.dict(
                "backend.copilot.sdk.env.os.environ",
                {
                    "NO_PROXY": "internal.example,metadata.internal,localhost",
                },
                clear=True,
            ),
        ):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(
                codex_gateway_url="http://localhost:43210",
                codex_gateway_token="loopback-capability",
            )

        assert set(result["NO_PROXY"].split(",")) == {
            "internal.example",
            "metadata.internal",
            "127.0.0.1",
            "localhost",
            "::1",
        }
        assert result["no_proxy"] == result["NO_PROXY"]

    def test_codex_override_requires_url_and_token_together(self):
        cfg = _make_config(use_openrouter=False)
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            with pytest.raises(ValueError, match="must be provided together"):
                build_sdk_env(codex_gateway_url="http://127.0.0.1:43210")

    def test_codex_override_rejects_non_loopback_url(self):
        cfg = _make_config(use_openrouter=False)
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            with pytest.raises(ValueError, match="loopback HTTP URL"):
                build_sdk_env(
                    codex_gateway_url="https://example.com",
                    codex_gateway_token="must-not-leak",
                )


class TestAutocompactPctSonnet5Scaling:
    """Sonnet 5's trigger percentage is scaled by the ~1.3x tokenizer
    inflation so compaction fires at the same text-equivalent point as on
    4.x models (50% -> 65% = 130K tokens of the pinned 200K window
    ~= 100K 4.x-tokens' worth)."""

    @pytest.mark.parametrize("model", ["anthropic/claude-sonnet-5", "claude-sonnet-5"])
    def test_sonnet_5_scaled(self, model):
        cfg = _make_config(
            use_openrouter=False, claude_agent_autocompact_pct_override=50
        )
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model=model)

        assert result.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE") == "65"

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-sonnet-4-5",  # substring near-miss guard
            "anthropic/claude-opus-4-7",
            "anthropic/claude-opus-4-8",
        ],
    )
    def test_non_sonnet_5_not_scaled(self, model):
        cfg = _make_config(
            use_openrouter=False, claude_agent_autocompact_pct_override=50
        )
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model=model)

        assert result.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE") == "50"

    def test_scaled_value_capped_below_cli_ceiling(self):
        cfg = _make_config(
            use_openrouter=False, claude_agent_autocompact_pct_override=80
        )
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model="anthropic/claude-sonnet-5")

        assert result.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE") == "90"


class TestContextWindowPin:
    """The window the CLI compacts against is ours, not the CLI's guess.

    At or below the default, ``CLAUDE_CODE_DISABLE_1M_CONTEXT`` holds the model
    window itself at 200K too, making the default a real cap.  On Moonshot
    routes the pin is capped at the catalog's real window for the SKU.
    """

    @pytest.mark.parametrize(
        "window, expected, kill_switch",
        [
            # At the 200K default and a 50% override the CLI compacts at ~90K
            # (~117K on Sonnet 5, whose override is scaled to 65%).
            (200_000, "200000", True),
            (200_001, "200001", False),
            (1_000_000, "1000000", False),
        ],
    )
    def test_window_pin_and_1m_kill_switch(self, window, expected, kill_switch):
        """Above the default the kill-switch has to come off with the raise:
        it would otherwise clamp the model window back to 200K and swallow it."""
        cfg = _make_config(use_openrouter=False, claude_agent_context_window=window)
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model="anthropic/claude-sonnet-5")

        assert result.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == expected
        assert ("CLAUDE_CODE_DISABLE_1M_CONTEXT" in result) is kill_switch

    @pytest.mark.parametrize(
        "model, window, expected, kill_switch",
        [
            # Kimi K2.x really serves 262,144: a 1M pin would put the trigger
            # at ~967K, past the point the provider rejects the request.
            ("moonshotai/kimi-k2.5", 1_000_000, "262144", False),
            ("moonshotai/kimi-k2-thinking", 500_000, "262144", False),
            # K3 really does serve 1M, so the raise survives there.
            ("moonshotai/kimi-k3", 1_000_000, "1000000", False),
            # A SKU the catalog does not carry falls back to the window the
            # CLI assumes anyway, rather than to an unbounded raise.
            ("moonshotai/kimi-k3.0", 1_000_000, "200000", True),
            # At or below the real window the configured pin is untouched.
            ("moonshotai/kimi-k2.5", 262_144, "262144", False),
            ("moonshotai/kimi-k2.5", 200_000, "200000", True),
        ],
    )
    def test_pin_capped_at_moonshot_real_window(
        self, model, window, expected, kill_switch
    ):
        """The CLI's model table has no Moonshot entry, so the pin is the only
        thing holding the trigger inside Kimi's real window."""
        cfg = _make_config(
            use_openrouter=True,
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
            thinking_standard_model=model,
            claude_agent_context_window=window,
        )
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model=model)

        assert result.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == expected
        assert ("CLAUDE_CODE_DISABLE_1M_CONTEXT" in result) is kill_switch

    def test_context_window_rejects_out_of_range(self):
        """Pydantic bounds (ge=100_000, le=1_000_000) are the only guard between
        a typo'd env var and the CLI's own clamps."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _make_config(claude_agent_context_window=99_999)
        with pytest.raises(ValidationError):
            _make_config(claude_agent_context_window=1_000_001)

    def test_window_defaults_to_none(self):
        """Unset means the route's engine default, not 200K everywhere."""
        assert _make_config().claude_agent_context_window is None

    def test_direct_anthropic_defaults_to_claude_engine_window(self):
        """The operator-keyed Anthropic route runs Claude models: 1M, no gate."""
        cfg = _make_config(use_openrouter=False)
        assert cfg.transport.name == "direct_anthropic"
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model="anthropic/claude-sonnet-5")

        assert result.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == "1000000"
        assert "CLAUDE_CODE_DISABLE_1M_CONTEXT" not in result

    @patch("backend.copilot.sdk.env.validate_subscription")
    def test_subscription_pins_200k_to_protect_the_plan_limit(self, _mock_validate):
        """Subscription turns draw on the subscriber's own plan: a 700K chat
        resends 700K every turn, so the route is held to 200K with the 1M
        gate set even though the engine would run 1M."""
        cfg = _make_config(use_claude_code_subscription=True)
        assert cfg.transport.name == "subscription"
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model="anthropic/claude-sonnet-5")

        assert result.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == "200000"
        assert result.get("CLAUDE_CODE_DISABLE_1M_CONTEXT") == "1"

    def test_openrouter_platform_default_stays_200k(self):
        """The platform route keeps today's behaviour: 200K pin, gate set."""
        cfg = _make_config(
            use_openrouter=True,
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
        )
        assert cfg.transport.name == "openrouter"
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model="anthropic/claude-sonnet-5")

        assert result.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == "200000"
        assert result.get("CLAUDE_CODE_DISABLE_1M_CONTEXT") == "1"

    def test_openrouter_moonshot_default_capped_at_sku_window(self):
        """Kimi K2.5 really serves 262,144 — below the platform pin, so the
        lower of the two (the pin) applies and the gate stays set."""
        cfg = _make_config(
            use_openrouter=True,
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
            thinking_standard_model="moonshotai/kimi-k2.5",
        )
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model="moonshotai/kimi-k2.5")

        assert result.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == "200000"
        assert result.get("CLAUDE_CODE_DISABLE_1M_CONTEXT") == "1"

    @pytest.mark.parametrize(
        "config_overrides, env_kwargs, model",
        [
            ({"use_openrouter": False}, {}, "anthropic/claude-sonnet-5"),
            (
                {"use_claude_code_subscription": True},
                {},
                "anthropic/claude-sonnet-5",
            ),
            (
                {
                    "use_openrouter": True,
                    "api_key": "sk-or-test",
                    "base_url": "https://openrouter.ai/api/v1",
                },
                {},
                "anthropic/claude-sonnet-5",
            ),
            (
                {
                    "use_openrouter": True,
                    "api_key": "sk-or-test",
                    "base_url": "https://openrouter.ai/api/v1",
                },
                {
                    "codex_gateway_url": "http://127.0.0.1:9",
                    "codex_gateway_token": "codex-test-token",
                },
                "gpt-6-astra",
            ),
        ],
    )
    @patch("backend.copilot.sdk.env.validate_subscription")
    def test_explicit_window_wins_on_every_route(
        self, _mock_validate, config_overrides, env_kwargs, model
    ):
        """An operator pin beats every engine default, Codex included."""
        cfg = _make_config(claude_agent_context_window=300_000, **config_overrides)
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model=model, **env_kwargs)

        assert result.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == "300000"
        assert "CLAUDE_CODE_DISABLE_1M_CONTEXT" not in result


class TestCodexRouteContext:
    """The Codex route mirrors the Codex engine: 272K window, 90% trigger.

    The gateway speaks for the connected account, so the deployment-wide
    profile — including ``local`` — is bypassed, not read.
    """

    _GATEWAY_KWARGS = {
        "codex_gateway_url": "http://127.0.0.1:9",
        "codex_gateway_token": "codex-test-token",
    }

    def _codex_config(self, **overrides):
        defaults = {
            "use_openrouter": True,
            "api_key": "sk-or-test",
            "base_url": "https://openrouter.ai/api/v1",
        }
        defaults.update(overrides)
        return _make_config(**defaults)

    def test_codex_route_pins_the_account_window(self):
        """When the account advertises a window for the routed model, the
        pin, the ceiling and the trigger all follow it."""
        cfg = self._codex_config()
        engine = CodexEngineWindow(
            context_window=400_000, auto_compact_token_limit=200_000
        )
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(
                model="gpt-6-astra", codex_engine=engine, **self._GATEWAY_KWARGS
            )

        assert result.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == "400000"
        assert result.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS") == "400000"
        assert result.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE") == "50"
        assert "CLAUDE_CODE_DISABLE_1M_CONTEXT" not in result

    @pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-5.6-terra", "gpt-5.6-sol"])
    def test_codex_route_pins_engine_default(self, model):
        cfg = self._codex_config()
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model=model, **self._GATEWAY_KWARGS)

        assert result.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == "272000"
        assert "CLAUDE_CODE_DISABLE_1M_CONTEXT" not in result
        assert result.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE") == "90"

    def test_codex_route_wins_over_local_profile(self):
        """The gateway speaks Anthropic's wire protocol even when the
        configured transport doesn't — the local profile is bypassed."""
        cfg = _make_config(
            use_local=True,
            api_key="ollama",
            base_url="http://host:11434/v1",
        )
        assert cfg.transport.name == "local"
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model="gpt-5.6-terra", **self._GATEWAY_KWARGS)

        assert result.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == "272000"
        assert "CLAUDE_CODE_DISABLE_1M_CONTEXT" not in result
        assert result.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE") == "90"

    def test_codex_unlisted_slug_falls_back_to_engine_default(self):
        """Unknown slugs take 272K too — the same fallback codex-rs uses."""
        cfg = self._codex_config()
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model="gpt-9.9-zzz", **self._GATEWAY_KWARGS)

        assert result.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == "272000"
        assert "CLAUDE_CODE_DISABLE_1M_CONTEXT" not in result
        assert result.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE") == "90"

    def test_codex_moonshot_slug_still_gets_engine_trigger(self):
        """A moonshot-shaped slug over the gateway runs on Codex infra, not
        the Moonshot endpoint — the Moonshot skip must not apply."""
        cfg = self._codex_config()
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model="moonshotai/kimi-k2.5", **self._GATEWAY_KWARGS)

        assert result.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == "272000"
        assert result.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE") == "90"

    def test_codex_zero_pct_still_omits_override(self):
        """The 0 kill-switch omits the trigger var on the Codex route too."""
        cfg = self._codex_config(claude_agent_autocompact_pct_override=0)
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model="gpt-6-astra", **self._GATEWAY_KWARGS)

        assert result.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == "272000"
        assert "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE" not in result

    def test_codex_route_lifts_the_model_window_assumption(self):
        """The pin is clamped to the window the CLI assumes for an
        unrecognised slug, so it only bites once MAX_CONTEXT_TOKENS moves
        that assumption.  Measured on CLI 2.1.274: without this, pins of
        200K/272K/1M produce identical compaction schedules."""
        cfg = self._codex_config()
        with patch("backend.copilot.sdk.env.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(model="gpt-6-astra", **self._GATEWAY_KWARGS)

        assert result.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS") == "272000"
        assert result["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == (
            result["CLAUDE_CODE_AUTO_COMPACT_WINDOW"]
        )

    def test_non_codex_routes_leave_the_model_table_alone(self):
        """Off the Codex route the CLI knows the model, and raising its
        ceiling would push the client-side length guard past what the
        provider accepts — a provider 400 mid-turn instead of a clean
        local refusal."""
        with patch("backend.copilot.sdk.env.config", _make_config()):
            from backend.copilot.sdk.env import build_sdk_env

            result = build_sdk_env(
                session_id="s1", user_id="u1", sdk_cwd="/tmp", model="claude-opus-4-8"
            )

        assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in result


class TestDescribeSdkContext:
    def test_codex_route_summary(self):
        from backend.copilot.sdk.env import describe_sdk_context

        line = describe_sdk_context(
            route="codex",
            model="gpt-6-astra",
            sdk_env={
                "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "272000",
                "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "90",
            },
        )
        assert line == (
            "route=codex model=gpt-6-astra window=272000 "
            "trigger_pct=90 disable_1m_context=false"
        )

    def test_platform_route_with_kill_switch_and_default_trigger(self):
        from backend.copilot.sdk.env import describe_sdk_context

        line = describe_sdk_context(
            route="openrouter",
            model="anthropic/claude-sonnet-4-6",
            sdk_env={
                "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "200000",
                "CLAUDE_CODE_DISABLE_1M_CONTEXT": "1",
            },
        )
        assert line == (
            "route=openrouter model=anthropic/claude-sonnet-4-6 window=200000 "
            "trigger_pct=<cli-default> disable_1m_context=true"
        )

    def test_never_leaks_secrets(self):
        from backend.copilot.sdk.env import describe_sdk_context

        line = describe_sdk_context(
            route="openrouter",
            model="m",
            sdk_env={
                "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "200000",
                "ANTHROPIC_AUTH_TOKEN": "sk-or-very-secret",
                "ANTHROPIC_CUSTOM_HEADERS": "x-user-id: u-secret",
            },
        )
        assert "sk-or-very-secret" not in line
        assert "u-secret" not in line
