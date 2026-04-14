"""Tests for build_sdk_env() — the SDK subprocess environment builder."""

from unittest.mock import patch

import pytest

from backend.copilot.config import ChatConfig

# ---------------------------------------------------------------------------
# Helpers — build a ChatConfig with explicit field values so tests don't
# depend on real environment variables.
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> ChatConfig:
    """Create a ChatConfig with safe defaults, applying *overrides*."""
    defaults = {
        "use_claude_code_subscription": False,
        "use_openrouter": False,
        "api_key": None,
        "base_url": None,
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
