"""Which credential a Claude Code CLI launch carries, and where it may go.

Claude OAuth subscriptions are for self-hosted, local-only use.  The CLI
sends whatever credential it finds to whatever ``ANTHROPIC_BASE_URL`` it is
given, with no host check of its own, and the SDK merges a launch's ``env``
over the parent process environment.  So every launch states its credential
and base URL explicitly: a key-based launch blanks every Claude OAuth
variable and never inherits a base URL, and the subscription launch may only
talk to Anthropic or this machine.

Environment variables cannot stop the CLI from reading a ``claude login``
credentials file already on disk (``$CLAUDE_CONFIG_DIR/.credentials.json``,
or the macOS keychain).  The CLI only falls back to it when no API key or
auth token is set, so every key-based launch here carries a non-empty one.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")

# Every variable the CLI reads a Claude OAuth credential from, plus the
# refresh token ``sdk/subscription.py`` provisions the credentials file from.
CLAUDE_OAUTH_BLANK: dict[str, str] = {
    "CLAUDE_CODE_OAUTH_TOKEN": "",
    "CLAUDE_CODE_OAUTH_REFRESH_TOKEN": "",
    "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR": "",
    "CLAUDE_CODE_REFRESH_TOKEN": "",
}


# The variables a host may hold its Claude login in instead of a
# ``claude login`` credentials file.
_HOST_LOGIN_VARIABLES = ("CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_OAUTH_REFRESH_TOKEN")


def subscription_env() -> dict[str, str]:
    """No key or token, so the CLI uses the host's Claude login, and no base
    URL, so it sends that login to Anthropic. A login the host holds in a
    variable rather than a credentials file travels with the launch, which
    keeps nothing else of the parent's (``launch_env.py``)."""
    env = {
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_AUTH_TOKEN": "",
        "ANTHROPIC_BASE_URL": "",
    }
    for name in _HOST_LOGIN_VARIABLES:
        if value := os.environ.get(name):
            env[name] = value
    return env


def require_loopback_http(url: str, what: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in LOOPBACK_HOSTS:
        raise ValueError(f"{what} must use a loopback HTTP URL")


def require_local_oauth_base_url(env: dict[str, str]) -> None:
    """Hold a Claude OAuth launch to Anthropic (empty) or a loopback URL.

    A variable absent from *env* is inherited from the parent process.
    """
    base_url = env.get("ANTHROPIC_BASE_URL", os.environ.get("ANTHROPIC_BASE_URL"))
    if base_url:
        require_loopback_http(base_url, "The Claude subscription route")


def direct_anthropic_env(api_key: str | None) -> dict[str, str]:
    """An Anthropic API key, sent as ``x-api-key`` to api.anthropic.com."""
    if not api_key:
        raise RuntimeError(
            "No API key configured: the direct_anthropic route needs "
            "CHAT_ANTHROPIC_API_KEY."
        )
    return {
        "ANTHROPIC_API_KEY": api_key,
        "ANTHROPIC_AUTH_TOKEN": "",
        "ANTHROPIC_BASE_URL": "",
        **CLAUDE_OAUTH_BLANK,
    }
