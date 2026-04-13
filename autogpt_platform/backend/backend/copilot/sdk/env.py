"""SDK environment variable builder — importable without circular deps.

Extracted from ``service.py`` so that ``backend.blocks.orchestrator``
can reuse the same subscription / OpenRouter / direct-Anthropic logic
without pulling in the full copilot service module (which would create a
circular import through ``executor`` → ``credit`` → ``block_cost_config``).
"""

from __future__ import annotations

import re

from backend.copilot.config import ChatConfig
from backend.copilot.sdk.subscription import validate_subscription

# ChatConfig is stateless (reads env vars) — a separate instance is fine.
# A singleton would require importing service.py which causes the circular dep
# this module was created to avoid.
config = ChatConfig()

# RFC 7230 §3.2.6 — keep only printable ASCII; strip control chars and non-ASCII.
_HEADER_SAFE_RE = re.compile(r"[^\x20-\x7e]")
_MAX_HEADER_VALUE_LEN = 128


def build_sdk_env(
    session_id: str | None = None,
    user_id: str | None = None,
    sdk_cwd: str | None = None,
) -> dict[str, str]:
    """Build env vars for the SDK CLI subprocess.

    Three modes (checked in order):
    1. **Subscription** — clears all keys; CLI uses ``claude login`` auth.
    2. **Direct Anthropic** — subprocess inherits ``ANTHROPIC_API_KEY``
       from the parent environment (no overrides needed).
    3. **OpenRouter** (default) — overrides base URL and auth token to
       route through the proxy, with Langfuse trace headers.

    All modes receive workspace isolation (``CLAUDE_CODE_TMPDIR``) and
    security hardening env vars to prevent .claude.md loading, prompt
    history persistence, auto-memory writes, and non-essential traffic.
    """
    # --- Mode 1: Claude Code subscription auth ---
    if config.use_claude_code_subscription:
        validate_subscription()
        env: dict[str, str] = {
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": "",
            "ANTHROPIC_BASE_URL": "",
        }

    # --- Mode 2: Direct Anthropic (no proxy hop) ---
    elif not config.openrouter_active:
        # Clear OAuth tokens so CLI uses ANTHROPIC_API_KEY from parent env
        # rather than subscription auth if the container has those tokens set.
        env = {
            "CLAUDE_CODE_OAUTH_TOKEN": "",
            "CLAUDE_CODE_REFRESH_TOKEN": "",
        }

    # --- Mode 3: OpenRouter proxy ---
    else:
        base = (config.base_url or "").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        env = {
            "ANTHROPIC_BASE_URL": base,
            "ANTHROPIC_AUTH_TOKEN": config.api_key or "",
            "ANTHROPIC_API_KEY": "",  # force CLI to use AUTH_TOKEN
            "CLAUDE_CODE_OAUTH_TOKEN": "",  # prevent OAuth override of ANTHROPIC_AUTH_TOKEN
            "CLAUDE_CODE_REFRESH_TOKEN": "",  # prevent token refresh via subscription
        }

        # Inject broadcast headers so OpenRouter forwards traces to Langfuse.
        def _safe(v: str) -> str:
            return _HEADER_SAFE_RE.sub("", v).strip()[:_MAX_HEADER_VALUE_LEN]

        parts = []
        if session_id:
            parts.append(f"x-session-id: {_safe(session_id)}")
        if user_id:
            parts.append(f"x-user-id: {_safe(user_id)}")
        if parts:
            env["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(parts)

    # --- Common: workspace isolation + security hardening (all modes) ---
    # Route subagent temp files into the per-session workspace so output
    # files are accessible (fixes /tmp/claude-0/ permission errors in E2B).
    if sdk_cwd:
        env["CLAUDE_CODE_TMPDIR"] = sdk_cwd

    # Harden multi-tenant deployment: prevent loading untrusted workspace
    # .claude.md files, writing auto-memory, and sending non-essential
    # telemetry traffic.
    env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

    return env
