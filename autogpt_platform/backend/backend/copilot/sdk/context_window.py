"""Per-route SDK context window + compaction trigger resolution.

``build_sdk_env`` pins the CLI's perceived window explicitly
(``CLAUDE_CODE_AUTO_COMPACT_WINDOW``) so the compaction trigger is a
deliberate value rather than an emergent property of whichever bundled CLI
we ship. Each route is held to its coding engine's default window (or the
engine max where no default is published); the platform (openrouter) route
keeps the CLI's own 200K default.

Lives here rather than in ``env.py`` to keep that module under the
300-line guideline. Import-safe: only depends on ``config`` (constants)
and ``moonshot`` (which reads the immutable catalog snapshot) — the same
leaves ``env.py`` already imports, so no new import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.copilot.config import (
    CLI_DEFAULT_CONTEXT_WINDOW,
    CODEX_ENGINE_AUTOCOMPACT_PCT,
    CODEX_ENGINE_CONTEXT_WINDOW,
)
from backend.copilot.moonshot import is_moonshot_model, moonshot_context_window

if TYPE_CHECKING:
    from backend.copilot.config import ChatConfig


def pinned_context_window(
    config: ChatConfig,
    model: str | None,
    *,
    codex_route: bool,
    local_window: int | None = None,
) -> int:
    """Window (tokens) to pin the SDK subprocess to on this route.

    Precedence: an explicit ``claude_agent_context_window`` wins
    everywhere; otherwise the Codex route takes the Codex engine default,
    the local route takes the probed backend window (``local_window``,
    falling back to the profile's blind constant when the caller has
    none), and every other route takes its transport profile's engine
    default. Moonshot routes then take the lower of that pin and the
    SKU's real window — a pin past the point the provider rejects the
    turn would put the compaction trigger where it never fires. Anthropic
    routes take the pin as given: the catalog holds every Anthropic entry
    at 200K pending the Claude-5 tokenizer soak, which would cancel a
    legitimate raise.

    The Codex branch must not read ``config.transport``: a connected
    Codex account wins over the deployment-wide profile, including
    ``local``. An unlisted Codex SKU (or ``model=None``) takes the engine
    default too, exactly like codex-rs itself.
    """
    if config.claude_agent_context_window is not None:
        window = config.claude_agent_context_window
    elif codex_route:
        window = CODEX_ENGINE_CONTEXT_WINDOW
    elif config.transport.name == "local":
        window = (
            local_window
            if local_window is not None
            else config.transport.sdk_context_window
        )
    else:
        window = config.transport.sdk_context_window
    if not codex_route and is_moonshot_model(model):
        window = min(
            window, moonshot_context_window(model) or CLI_DEFAULT_CONTEXT_WINDOW
        )
    return window


# Sonnet 5's tokenizer counts ~30% more tokens than 4.x for the same text,
# so the same 50%-of-200K trigger would compact at ~77% of the *text* budget
# 4.x sessions get.  Scaling the trigger by the inflation factor keeps the
# effective text-equivalent context at parity (50% -> 65% = 130K tokens
# ~= 100K 4.x-tokens' worth), without touching the perceived window.
_SONNET_5_TOKENIZER_INFLATION = 1.3


def autocompact_pct(config: ChatConfig, model: str | None, *, codex_route: bool) -> int:
    """Auto-compaction trigger percentage for this route and ``model``.

    The Codex route mirrors the engine's own 90%-of-window trigger (its
    costs accrue to the connected ChatGPT account, so the Anthropic
    cache-cost rationale behind the configured default does not apply
    there). A configured 0 still omits the override on every route. The
    local route returns 0 (no override: the operator's backend has no
    Anthropic cache costs to cap, so the CLI default applies).
    Elsewhere the base value comes from config; Sonnet 5 is scaled up by
    the tokenizer-inflation factor (capped at 90, below the CLI's ~93%
    internal ceiling) so its compaction fires at the same
    text-equivalent point as on 4.x models.
    """
    if codex_route:
        if config.claude_agent_autocompact_pct_override == 0:
            return 0
        return CODEX_ENGINE_AUTOCOMPACT_PCT
    if config.transport.name == "local":
        return 0
    pct = config.claude_agent_autocompact_pct_override
    if model and "claude-sonnet-5" in model:
        pct = min(round(pct * _SONNET_5_TOKENIZER_INFLATION), 90)
    return pct


__all__ = [
    "autocompact_pct",
    "pinned_context_window",
]
