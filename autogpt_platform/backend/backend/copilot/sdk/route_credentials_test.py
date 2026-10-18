"""Unit tests for the per-launch credential rules in ``route_credentials``."""

from unittest.mock import patch

import pytest

from .route_credentials import (
    CLAUDE_OAUTH_BLANK,
    direct_anthropic_env,
    require_local_oauth_base_url,
    require_loopback_http,
    subscription_env,
)

_NON_LOOPBACK = [
    "https://remote-gateway.example.com",
    "https://api.anthropic.com",
    "https://openrouter.ai/api",
    # Names the CLI was probed sending the OAuth bearer to: only the three
    # literal loopback hosts count.
    "http://gw.localhost:8080",
    "http://localtest.me:8080",
    "http://127.0.0.1.nip.io:8080",
    "https://127.0.0.1:8443",
]
_LOOPBACK = ["http://127.0.0.1:8099", "http://localhost:8099", "http://[::1]:1"]


class TestRequireLoopbackHttp:
    @pytest.mark.parametrize("url", _NON_LOOPBACK)
    def test_refuses(self, url):
        with pytest.raises(ValueError, match="Thing must use a loopback HTTP URL"):
            require_loopback_http(url, "Thing")

    @pytest.mark.parametrize("url", _LOOPBACK)
    def test_allows(self, url):
        require_loopback_http(url, "Thing")


class TestRequireLocalOAuthBaseUrl:
    """The subscription launch may talk to Anthropic (no base URL) or to this
    machine, never anywhere else."""

    @pytest.mark.parametrize("url", _NON_LOOPBACK)
    def test_refuses_a_remote_base_url(self, url):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Claude subscription route"):
                require_local_oauth_base_url({"ANTHROPIC_BASE_URL": url})

    @pytest.mark.parametrize("url", ["", *_LOOPBACK])
    def test_allows_anthropic_or_loopback(self, url):
        with patch.dict("os.environ", {}, clear=True):
            require_local_oauth_base_url({"ANTHROPIC_BASE_URL": url})

    def test_an_absent_base_url_is_read_from_the_parent(self):
        parent = {"ANTHROPIC_BASE_URL": "https://remote-gateway.example.com"}
        with patch.dict("os.environ", parent, clear=True):
            with pytest.raises(ValueError, match="Claude subscription route"):
                require_local_oauth_base_url({})

    def test_an_explicit_blank_overrides_the_parent(self):
        parent = {"ANTHROPIC_BASE_URL": "https://remote-gateway.example.com"}
        with patch.dict("os.environ", parent, clear=True):
            require_local_oauth_base_url({"ANTHROPIC_BASE_URL": ""})


def test_subscription_env_sets_no_key_and_no_base_url():
    with patch.dict("os.environ", {}, clear=True):
        assert subscription_env() == {
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": "",
            "ANTHROPIC_BASE_URL": "",
        }


def test_a_host_login_held_in_variables_travels_with_the_launch():
    """The launch keeps nothing else of the parent's, so the host's own
    login goes with it; an integration key does not."""
    parent = {
        "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-FAKE",
        "CLAUDE_CODE_OAUTH_REFRESH_TOKEN": "FAKE-refresh",
        "ANTHROPIC_API_KEY": "sk-ant-FAKE-integration",
    }
    with patch.dict("os.environ", parent, clear=True):
        assert subscription_env() == {
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": "",
            "ANTHROPIC_BASE_URL": "",
            "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-FAKE",
            "CLAUDE_CODE_OAUTH_REFRESH_TOKEN": "FAKE-refresh",
        }


def test_oauth_blank_covers_every_variable_the_cli_reads():
    assert set(CLAUDE_OAUTH_BLANK) == {
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_REFRESH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
        "CLAUDE_CODE_REFRESH_TOKEN",
    }
    assert set(CLAUDE_OAUTH_BLANK.values()) == {""}


class TestDirectAnthropicEnv:
    def test_explicit_key_no_base_url_no_oauth(self):
        assert direct_anthropic_env("sk-ant-key") == {
            "ANTHROPIC_API_KEY": "sk-ant-key",
            "ANTHROPIC_AUTH_TOKEN": "",
            "ANTHROPIC_BASE_URL": "",
            **CLAUDE_OAUTH_BLANK,
        }

    @pytest.mark.parametrize("key", [None, ""])
    def test_refuses_without_a_key(self, key):
        """With no key the CLI would fall back to a ``claude login``
        credentials file on the host."""
        with pytest.raises(RuntimeError, match="CHAT_ANTHROPIC_API_KEY"):
            direct_anthropic_env(key)
