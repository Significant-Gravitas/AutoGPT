"""Compaction arithmetic over the window and trigger a launch is pinned to.

The route profile (``copilot/route_profile.py``) decides each route's window
and trigger; ``sdk/env.py`` pins the CLI to them, and every compressor the
copilot runs itself sizes its output from the same two numbers here. One
window per launch, one trigger, one set of budgets.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

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
        a trigger; the codex route then compacts at its default of 90%.
        """
        limit = self.auto_compact_token_limit
        if limit is None or limit <= 0 or limit >= self.context_window:
            return None
        return max(1, min(_MAX_TRIGGER_PCT, limit * 100 // self.context_window))


def cli_autocompact_threshold(window: int, pct: int | None) -> int:
    """Tokens at which the pinned CLI auto-compacts.

    Mirrors the bundled CLI's formula, ``min(window * pct/100, window - 13K)``;
    with no trigger the CLI's own leaves only the buffer.
    """
    if pct:
        return min(window * pct // 100, window - _CLI_COMPACT_BUFFER_TOKENS)
    return window - _CLI_COMPACT_BUFFER_TOKENS


def compaction_target_tokens(window: int, pct: int | None) -> int:
    """Output budget for the copilot's own compressors, pre-query and retry.

    The CLI's threshold less ``COMPACTION_HEADROOM_TOKENS``, floored so some
    history always survives.  A target derived from a different window than
    the pin is a second threshold authority: it either fires early forever or
    lands over the pin.
    """
    threshold = cli_autocompact_threshold(window, pct)
    return max(_COMPACTION_TARGET_FLOOR_TOKENS, threshold - COMPACTION_HEADROOM_TOKENS)


def retry_target_tokens(window: int) -> tuple[int, int]:
    """Budgets for the no-transcript fallback: first retry, then any later one.

    A quarter of the pinned window first (50K at 200K). The last budget is the
    bare-message floor itself, on every route: that is what makes the final
    retry send the message alone rather than fail "Prompt is too long".
    """
    return window // 4, BARE_MESSAGE_TOKEN_FLOOR


def seed_target_tokens(window: int) -> int:
    """Budget for seeding the transcript builder on a turn with no CLI session.

    Fifteen percent of the pinned window (30K at 200K), below the first retry
    budget so the seeded upload stays compact and later gap fills small.
    """
    return window * 3 // 20


__all__ = [
    "BARE_MESSAGE_TOKEN_FLOOR",
    "COMPACTION_HEADROOM_TOKENS",
    "CodexEngineWindow",
    "cli_autocompact_threshold",
    "compaction_target_tokens",
    "retry_target_tokens",
    "seed_target_tokens",
]
