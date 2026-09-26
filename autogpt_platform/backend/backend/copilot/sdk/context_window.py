"""Per-route SDK context window, compaction trigger, and compression budgets.

``build_sdk_env`` pins the CLI's perceived window explicitly
(``CLAUDE_CODE_AUTO_COMPACT_WINDOW``) so the compaction trigger is a
deliberate value rather than an emergent property of whichever bundled CLI
we ship — and every compressor the copilot runs itself sizes its output from
that same pin.  One window per route, one trigger, one set of numbers.

- direct_anthropic: the engine default, 1M.  The CLI clamps a pin *above*
  its own model-table window (measured, 2.1.274), so on this route the pin
  can only lower; 1M means "let the table decide".
- subscription: 200K.  The turns draw on the subscriber's plan, and a chat
  resending 700K per turn drains a usage window in a few messages.
- codex: what the connected account advertises for the routed model
  (``CodexEngineWindow``), falling back to the Codex engine default (272K,
  90% trigger) when the payload carries no window.  Either way
  ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` is raised alongside so the pin is not
  clamped at the 200K the CLI assumes for a slug it does not recognise.
- openrouter (platform): 200K, the CLI default — context past 200K is
  where Anthropic cache-creation cost dominates the bill.
- local: no SDK pin on this transport.

Lives here rather than in ``env.py`` to keep that module under the
300-line guideline. Import-safe: only depends on ``config`` (constants)
and ``moonshot`` (which reads the immutable catalog snapshot) — the same
leaves ``env.py`` already imports, so no new import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from backend.copilot.config import (
    CLI_DEFAULT_CONTEXT_WINDOW,
    CODEX_ENGINE_AUTOCOMPACT_PCT,
    CODEX_ENGINE_CONTEXT_WINDOW,
)
from backend.copilot.moonshot import is_moonshot_model, moonshot_context_window

if TYPE_CHECKING:
    from backend.copilot.config import ChatConfig

# The CLI keeps this much of the window back for the summary it writes.
_CLI_COMPACT_BUFFER_TOKENS = 13_000

# Headroom below the CLI's autocompact threshold when sizing the copilot's
# own compressors.  Without it the post-compaction context lands just under
# the threshold and the next assistant message tips it back over: the CLI
# re-compacts at once, and again, until its rapid-refill breaker ends the
# turn.
COMPACTION_HEADROOM_TOKENS = 20_000

# Some history must always survive a compaction, however small the window.
_COMPACTION_TARGET_FLOOR_TOKENS = 10_000

# Below this budget the context is so tight that injecting any history would
# likely exceed the limit whatever it contains; ``_build_query_message`` sends
# the bare message instead.  The last retry budget sits exactly here so a
# session whose stored history exceeds the window falls through to that
# escape hatch instead of exhausting every attempt (SENTRY-1207).
BARE_MESSAGE_TOKEN_FLOOR = 5_000

# The CLI's own trigger ceiling is ~93% of the window; an advertised limit
# that works out higher is held here so the override still takes effect.
_MAX_TRIGGER_PCT = 90


class CodexEngineWindow(BaseModel):
    """What the connected ChatGPT account advertises for the routed model.

    ``context_window`` is the window Codex itself runs the model at (the
    API maximum is larger); ``auto_compact_token_limit`` is the absolute
    token count at which codex-rs compacts, when the payload carries one.
    """

    model_config = ConfigDict(frozen=True)

    context_window: int
    auto_compact_token_limit: int | None = None

    @property
    def trigger_pct(self) -> int | None:
        """The advertised limit as a percentage of the window, or None.

        None when no limit was advertised, or when it is not below the
        window — codex-rs would never compact on such a value, so it is not
        a trigger; ``autocompact_pct`` then falls back to the engine default
        of 90%.
        """
        limit = self.auto_compact_token_limit
        if limit is None or limit <= 0 or limit >= self.context_window:
            return None
        return max(1, min(_MAX_TRIGGER_PCT, limit * 100 // self.context_window))


def pinned_context_window(
    config: ChatConfig,
    model: str | None,
    *,
    codex_route: bool,
    codex_engine: CodexEngineWindow | None = None,
) -> int:
    """Window (tokens) to pin the SDK subprocess to on this route.

    Precedence: an explicit ``claude_agent_context_window`` wins
    everywhere; otherwise the Codex route takes the account's advertised
    window for the routed model (``codex_engine``) or, without one, the
    Codex engine default, and every other route takes its transport
    profile's engine default.  Moonshot routes then take the lower of
    that pin and the SKU's real window — a pin past the point the
    provider rejects the turn would put the compaction trigger where it
    never fires.  Anthropic routes take the pin as given: the catalog
    holds every Anthropic entry at 200K pending the Claude-5 tokenizer
    soak, which would cancel a legitimate raise.

    The Codex branch must not read ``config.transport``: a connected
    Codex account wins over the deployment-wide profile, including
    ``local``.  An unlisted Codex SKU (or ``model=None``) takes the engine
    default too, exactly like codex-rs itself.
    """
    if config.claude_agent_context_window is not None:
        window = config.claude_agent_context_window
    elif codex_route:
        window = (
            codex_engine.context_window
            if codex_engine is not None
            else CODEX_ENGINE_CONTEXT_WINDOW
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


def autocompact_pct(
    config: ChatConfig,
    model: str | None,
    *,
    codex_route: bool,
    codex_engine: CodexEngineWindow | None = None,
) -> int:
    """Auto-compaction trigger percentage for this route and ``model``.

    The Codex route follows the account's advertised compaction limit for
    the routed model when there is one, and otherwise mirrors the engine's
    own 90%-of-window trigger (its costs accrue to the connected ChatGPT
    account, so the Anthropic cache-cost rationale behind the configured
    default does not apply there).  A configured 0 still omits the override
    on every route.  Elsewhere the base value comes from config; Sonnet 5
    is scaled up by the tokenizer-inflation factor (capped at 90, below the
    CLI's ~93% internal ceiling) so its compaction fires at the same
    text-equivalent point as on 4.x models.
    """
    if codex_route:
        if config.claude_agent_autocompact_pct_override == 0:
            return 0
        advertised = codex_engine.trigger_pct if codex_engine is not None else None
        return advertised if advertised is not None else CODEX_ENGINE_AUTOCOMPACT_PCT
    pct = config.claude_agent_autocompact_pct_override
    if model and "claude-sonnet-5" in model:
        pct = min(round(pct * _SONNET_5_TOKENIZER_INFLATION), 90)
    return pct


def cli_autocompact_threshold(
    config: ChatConfig,
    model: str | None,
    *,
    codex_route: bool,
    codex_engine: CodexEngineWindow | None = None,
) -> int:
    """Tokens at which the pinned subprocess will auto-compact.

    Mirrors the bundled CLI's formula, ``min(window * pct/100, window -
    13K)``, against the same window and pct ``build_sdk_env`` pins.  The
    Codex route always runs the percentage trigger (a moonshot-shaped slug
    there still runs on Codex infra); elsewhere a Moonshot route omits the
    override and takes the CLI's default trigger.
    """
    window = pinned_context_window(
        config, model, codex_route=codex_route, codex_engine=codex_engine
    )
    pct = autocompact_pct(
        config, model, codex_route=codex_route, codex_engine=codex_engine
    )
    if pct > 0 and (codex_route or not is_moonshot_model(model)):
        return min(window * pct // 100, window - _CLI_COMPACT_BUFFER_TOKENS)
    return window - _CLI_COMPACT_BUFFER_TOKENS


def compaction_target_tokens(
    config: ChatConfig,
    model: str | None,
    *,
    codex_route: bool,
    codex_engine: CodexEngineWindow | None = None,
) -> int:
    """Output budget for the copilot's own compressors, pre-query and retry.

    The CLI's threshold less ``COMPACTION_HEADROOM_TOKENS``, floored so some
    history always survives.  Every compressor the copilot runs reads this
    and never the model catalog: a target derived from a different window
    than the pin is a second threshold authority, and it either fires early
    forever or lands over the pin.
    """
    threshold = cli_autocompact_threshold(
        config, model, codex_route=codex_route, codex_engine=codex_engine
    )
    return max(_COMPACTION_TARGET_FLOOR_TOKENS, threshold - COMPACTION_HEADROOM_TOKENS)


def retry_target_tokens(
    config: ChatConfig,
    model: str | None,
    *,
    codex_route: bool,
    codex_engine: CodexEngineWindow | None = None,
) -> tuple[int, int]:
    """Budgets for the no-transcript fallback: first retry, then any later one.

    A quarter of the pinned window first — 50K at the 200K platform default,
    which is what it was as a constant, and scaling with the route's window
    instead of assuming 200K everywhere.  The last budget is the bare-message
    floor itself, on every route: that is what makes the final retry send
    the message alone rather than fail "Prompt is too long".
    """
    window = pinned_context_window(
        config, model, codex_route=codex_route, codex_engine=codex_engine
    )
    return window // 4, BARE_MESSAGE_TOKEN_FLOOR


def seed_target_tokens(
    config: ChatConfig,
    model: str | None,
    *,
    codex_route: bool,
    codex_engine: CodexEngineWindow | None = None,
) -> int:
    """Budget for seeding the transcript builder on a turn with no CLI session.

    Fifteen percent of the pinned window (30K at 200K), below the first retry
    budget so the seeded upload stays compact and later gap fills small.
    """
    window = pinned_context_window(
        config, model, codex_route=codex_route, codex_engine=codex_engine
    )
    return window * 3 // 20


__all__ = [
    "BARE_MESSAGE_TOKEN_FLOOR",
    "COMPACTION_HEADROOM_TOKENS",
    "CodexEngineWindow",
    "autocompact_pct",
    "cli_autocompact_threshold",
    "compaction_target_tokens",
    "pinned_context_window",
    "retry_target_tokens",
    "seed_target_tokens",
]
