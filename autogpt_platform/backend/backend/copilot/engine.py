"""Which execution engine runs a turn.

Lives apart from the processor so the answer can be read without importing
the engines themselves: the processor pulls in both services at module
level, and the API layer needs this decision to describe a connection
before any turn exists.
"""

import logging

from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)


async def resolve_use_sdk(
    user_id: str | None,
    *,
    use_claude_code_subscription: bool,
    config_default: bool,
    thinking_available: bool = True,
) -> bool:
    """Pick the SDK vs baseline engine.

    Entirely the server's call: the Claude Code subscription override, then
    the ``COPILOT_SDK`` LaunchDarkly flag, then the config default. Callers
    used to be able to name an engine per request; nothing can now, so this
    is one decision rather than a request overriding it — which is also what
    makes it answerable ahead of a turn.

    ``thinking_available`` is the kill-switch for deployments where the SDK
    transport simply cannot run (today: ``CHAT_USE_LOCAL=true`` — Ollama
    doesn't speak Anthropic's wire protocol).
    """
    if not thinking_available:
        return False
    return use_claude_code_subscription or await is_feature_enabled(
        Flag.COPILOT_SDK,
        user_id or "anonymous",
        default=config_default,
    )
