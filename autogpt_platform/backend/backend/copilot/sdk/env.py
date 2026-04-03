"""SDK environment variable builder — importable without circular deps.

Extracted from ``service.py`` so that ``backend.blocks.orchestrator``
can reuse the same subscription / OpenRouter / direct-Anthropic logic
without pulling in the full copilot service module (which would create a
circular import through ``executor`` → ``credit`` → ``block_cost_config``).
"""

from __future__ import annotations

from backend.copilot.config import ChatConfig
from backend.copilot.sdk.subscription import validate_subscription

# ChatConfig is stateless (reads env vars) — a separate instance is fine.
# A singleton would require importing service.py which causes the circular dep
# this module was created to avoid.
config = ChatConfig()


def build_sdk_env(
    session_id: str | None = None,
    user_id: str | None = None,
    sdk_cwd: str | None = None,
) -> dict[str, str]:
    """Build env vars for the SDK CLI subprocess.

    Three modes (checked in order):
    1. **Subscription** — clears all keys; CLI uses ``claude login`` auth.
    2. **Direct Anthropic** — returns ``{}``; subprocess inherits
       ``ANTHROPIC_API_KEY`` from the parent environment.
    3. **OpenRouter** (default) — overrides base URL and auth token to
       route through the proxy, with Langfuse trace headers.

    When *sdk_cwd* is provided, ``CLAUDE_CODE_TMPDIR`` is set so that
    the CLI writes temp/sub-agent output inside the per-session workspace
    directory rather than an inaccessible system temp path.
    """
    # --- Mode 1: Claude Code subscription auth ---
    if config.use_claude_code_subscription:
        validate_subscription()
        env: dict[str, str] = {
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": "",
            "ANTHROPIC_BASE_URL": "",
        }
        if sdk_cwd:
            env["CLAUDE_CODE_TMPDIR"] = sdk_cwd
        return env

    # --- Mode 2: Direct Anthropic (no proxy hop) ---
    if not config.openrouter_active:
        env = {}
        if sdk_cwd:
            env["CLAUDE_CODE_TMPDIR"] = sdk_cwd
        return env

    # --- Mode 3: OpenRouter proxy ---
    base = (config.base_url or "").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    env = {
        "ANTHROPIC_BASE_URL": base,
        "ANTHROPIC_AUTH_TOKEN": config.api_key or "",
        "ANTHROPIC_API_KEY": "",  # force CLI to use AUTH_TOKEN
    }

    # Inject broadcast headers so OpenRouter forwards traces to Langfuse.
    def _safe(v: str) -> str:
        return v.replace("\r", "").replace("\n", "").strip()[:128]

    parts = []
    if session_id:
        parts.append(f"x-session-id: {_safe(session_id)}")
    if user_id:
        parts.append(f"x-user-id: {_safe(user_id)}")
    if parts:
        env["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(parts)

    if sdk_cwd:
        env["CLAUDE_CODE_TMPDIR"] = sdk_cwd

    return env
