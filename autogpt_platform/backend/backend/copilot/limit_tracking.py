"""``chat_blocked``: a usage limit refused, paused or queued someone's work.

One backend event every time a limit stops work, from whichever surface the
work came: the chat, a schedule, a chat bot, a workflow block, voice, the
external API, or background work such as subtasks and the nightly dream pass.
The event is sent where the refusal is answered, not where the limit is read,
because some reads only probe the allowance (offering "Resume at reset", for
one) and only the caller knows the surface.

Sending never raises and never waits: ``posthog_client.capture`` queues the
event for the client's background thread, and anything that fails here is
logged and dropped, so a refusal answers exactly as it would without it.

No message content and no personal data: the distinct id is the user id, and
the properties say only which limit, what happened, and where the account
stood against it.
"""

import functools
import logging
from contextvars import ContextVar
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Literal, ParamSpec

from pydantic import BaseModel

from backend.copilot.llm_attribution import current_turn_origin
from backend.copilot.tracking import SOURCE
from backend.util import posthog_client
from backend.util.posthog_events import PostHogEvent

if TYPE_CHECKING:
    from backend.copilot.active_turns import ConcurrentTurnLimitError
    from backend.copilot.rate_limit import RateLimitExceeded
    from backend.copilot.turn_pause import TurnPause
    from backend.copilot.turn_queue import InflightCapExceeded, TurnRefusal

logger = logging.getLogger(__name__)

BlockedLimit = Literal["daily", "weekly", "trial", "task", "rate", "concurrency"]
BlockedOutcome = Literal["refused", "paused", "queued"]
BlockedSurface = Literal[
    "chat", "schedule", "chat_bot", "block", "voice", "api", "background"
]

_RATE_LIMITS: frozenset[BlockedLimit] = frozenset({"rate", "concurrency"})

# The surface of the turn running now, as the executor worked it out; read by
# checks deep inside the turn, such as a tool's platform spend.
_turn_surface: ContextVar["BlockedSurface | None"] = ContextVar(
    "_turn_surface", default=None
)

_P = ParamSpec("_P")


def _never_raises(send: Callable[_P, None]) -> Callable[_P, None]:
    """Log and drop anything that goes wrong while sending, reading the
    refusal included, so the refusal answers as it would without PostHog."""

    @functools.wraps(send)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> None:
        try:
            send(*args, **kwargs)
        except Exception:
            logger.warning("Failed to send chat_blocked", exc_info=True)

    return guarded


# What started the running turn (``turn_tracing.turn_origin``) as a surface.
_SURFACE_FOR_ORIGIN: dict[str, BlockedSurface] = {
    "chat": "chat",
    "scheduled": "schedule",
    "automation": "block",
    "spawned": "background",
    "system": "background",
}


class LimitReading(BaseModel):
    """Where the account stood against the limit that stopped it, for analytics.

    ``used`` and ``cap`` are in the limit's unit: dollars for the cost limits,
    a count for the rate and concurrency caps.
    """

    tier: str | None = None
    used: float | None = None
    cap: float | None = None


def usd(microdollars: int | None) -> float | None:
    """Microdollars as dollars, for a reading."""
    return None if microdollars is None else round(microdollars / 1_000_000, 6)


def refills_at(
    limit: BlockedLimit, resets_at: datetime | None, reading: LimitReading | None
) -> datetime | None:
    """When a cost window refills: never for a trial's allowance (its
    ``resets_at`` is the trial's end), nor for a window that allows nothing."""
    if limit == "trial" or (reading is not None and reading.cap == 0):
        return None
    return resets_at


@_never_raises
def track_chat_blocked(
    user_id: str | None,
    *,
    limit: BlockedLimit,
    outcome: BlockedOutcome,
    surface: BlockedSurface,
    reading: LimitReading | None = None,
    resets_at: datetime | None = None,
) -> None:
    """Send one ``chat_blocked``. Never raises.

    ``resets_at`` is when the window refills; leave it out for a limit that
    doesn't (a trial's allowance, a window that allows nothing).
    """
    reading = reading or LimitReading()
    posthog_client.capture(
        user_id,
        PostHogEvent.CHAT_BLOCKED,
        {
            "reason": "rate_limit" if limit in _RATE_LIMITS else "usage_limit",
            "limit": limit,
            "outcome": outcome,
            "surface": surface,
            "tier": reading.tier,
            "used": reading.used,
            "cap": reading.cap,
            "resets_at": resets_at.isoformat() if resets_at else None,
        },
        source=SOURCE,
    )


@_never_raises
def track_limit_exceeded(
    user_id: str | None,
    exc: "RateLimitExceeded",
    *,
    surface: BlockedSurface,
    outcome: BlockedOutcome = "refused",
) -> None:
    """``chat_blocked`` for a cost limit ``check_rate_limit`` refused."""
    track_chat_blocked(
        user_id,
        limit=exc.window,
        outcome=outcome,
        surface=surface,
        reading=exc.reading,
        resets_at=exc.refills_at,
    )


@_never_raises
def track_turn_refusal(
    user_id: str | None,
    refusal: "TurnRefusal",
    *,
    surface: BlockedSurface,
    outcome: BlockedOutcome,
) -> None:
    """``chat_blocked`` for a ``turn_refusal`` that a usage window caused.

    A ban, a missing plan or Codex access, or an outage is not a usage limit,
    so it sends nothing.
    """
    if refusal.kind != "usage" or refusal.window is None:
        return
    track_chat_blocked(
        user_id,
        limit=refusal.window,
        outcome=outcome,
        surface=surface,
        reading=refusal.reading,
        resets_at=refills_at(refusal.window, refusal.resets_at, refusal.reading),
    )


@_never_raises
def track_turn_cap(
    user_id: str | None,
    exc: "ConcurrentTurnLimitError | InflightCapExceeded",
    *,
    surface: BlockedSurface,
    outcome: BlockedOutcome,
) -> None:
    """``chat_blocked`` for a turn the running or in-flight turn cap held back.

    ``queued`` when the turn waits for a free slot, ``refused`` when it doesn't.
    """
    track_chat_blocked(
        user_id,
        limit="concurrency",
        outcome=outcome,
        surface=surface,
        reading=LimitReading(used=exc.used, cap=exc.cap),
    )


@_never_raises
def track_turn_pause(
    user_id: str | None, pause: "TurnPause", *, surface: BlockedSurface
) -> None:
    """``chat_blocked`` for a turn that paused, or never started, on a limit.

    A step check-in is not a limit, so it sends nothing.
    """
    if pause.reason == "steps":
        return
    track_chat_blocked(
        user_id,
        limit=pause.reason,
        outcome="paused" if pause.started else "refused",
        surface=surface,
        reading=pause.reading,
        resets_at=refills_at(pause.reason, pause.resets_at, pause.reading),
    )


def set_turn_surface(surface: BlockedSurface | None) -> None:
    """Make *surface* the surface of the turn running now (the executor sets
    it for every turn, so a value never carries over)."""
    _turn_surface.set(surface)


def turn_surface(default: BlockedSurface = "chat") -> BlockedSurface:
    """The surface of the turn running now, or *default* outside a turn."""
    if (surface := _turn_surface.get()) is not None:
        return surface
    origin = current_turn_origin()
    if origin is None:
        return default
    return _SURFACE_FOR_ORIGIN.get(origin, default)


def executor_turn_surface(
    origin: str, *, scheduled: bool, source_platform: str | None
) -> BlockedSurface:
    """The surface of a turn the executor runs, from what started it: a
    schedule's turn or other unattended work, a chat bot's message or the
    web chat's, the AutoPilot block, or a subtask."""
    if origin == "scheduled":
        return "schedule" if scheduled else "background"
    if origin == "chat" and source_platform:
        return "chat_bot"
    return _SURFACE_FOR_ORIGIN.get(origin, "background")
