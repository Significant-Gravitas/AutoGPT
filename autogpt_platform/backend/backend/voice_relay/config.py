"""The voice relay's limits, prices and wire constants.

The caps are the product's limits for an interview before the paywall. The
relay enforces them and the browser only displays them; nothing a caller sends
can change them, and a test that needs shorter ones builds its own.
"""

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

# The person's microphone, as the call hears it, and Gemini Live's speech,
# both mono PCM16.
IN_RATE = 16_000
OUT_RATE = 24_000

# A frame of the person's audio past this size is dropped, not buffered.
MAX_FRAME_BYTES = 16_000
# Control messages from the browser are JSON text frames under this size.
MAX_TEXT_BYTES = 4_096
# The server's own frame limit. The control socket carries no audio: a frame
# under this reaches the relay, which drops what it cannot read, instead of
# closing the call.
WS_MAX_MESSAGE_BYTES = 16 * 1024
# How far ahead of real time a browser may push audio: a token bucket that
# refills at real time and holds this many seconds.
BURST_S = 2.0

# A socket must name its call this soon after it opens.
START_TIMEOUT_S = 5.0
# A start token's life, and how long a spent one is remembered (longer, so
# it cannot be spent twice).
TOKEN_TTL_S = 60
TOKEN_SPENT_TTL_S = 90
# A resume ticket outlives the call cap by this much.
RESUME_GRACE_S = 60
# One live call per user. Longer than a call can run, so a call never
# outlives its lock, and short enough that a dead pod's lock clears itself.
USER_LOCK_TTL_S = 360
# A call's record in Redis.
CALL_RECORD_TTL_S = 24 * 3600
# A live call's entry in the platform-wide count is refreshed this often, and
# counts as gone once it is this stale (its pod died).
LIVE_CALL_REFRESH_S = 30.0
LIVE_CALL_STALE_S = 120.0
# How long an "end that call" request waits for the pod that owns the call.
END_REQUEST_TTL_S = 600

# The call's room. Each call has a LiveKit room of its own, which carries its
# audio, joined by the person's browser and by the relay as Otto, and by no
# one else.
ROOM_PREFIX = "voice-"
RELAY_IDENTITY = "otto"
PERSON_IDENTITY_PREFIX = "user:"
ROOM_MAX_PARTICIPANTS = 2
# A room nobody joins closes after this long, and one everybody left after
# this long.
ROOM_EMPTY_TIMEOUT_S = 60
ROOM_DEPARTURE_TIMEOUT_S = 20
# The browser's room token. It only lets the browser into the room, and once
# in, LiveKit keeps it there for the whole call, handing it fresh tokens that
# live ten minutes. Removing the person revokes their tokens for ten minutes
# (ROOM_REVOCATION_S), so the minted one must never live longer than that:
# by then every token they could hold has expired.
ROOM_TOKEN_TTL_S = 600
ROOM_REVOCATION_S = 600
# LiveKit still takes a token this long past its expiry, and takes a
# revocation's cutoff up to this far ahead of its own clock.
ROOM_TOKEN_LEEWAY_S = 60
# A close revokes the person's tokens issued until this far past it, which
# also catches a token issued in the same second.
ROOM_REVOKE_AHEAD_S = 5
# The relay's own room token outlives the call cap by this much.
RELAY_ROOM_GRACE_S = 60
# The relay's join, and its leave, are given up past these. Each of the
# SDK's attempts at the connect ends inside the join's deadline.
ROOM_JOIN_TIMEOUT_S = 5.0
ROOM_CONNECT_TIMEOUT_S = 4.0
ROOM_LEAVE_TIMEOUT_S = 3.0
# Otto's speech is held for the room for at most this many seconds, the
# oldest dropped past it; once the call ends, what is held plays out for at
# most this long before the relay leaves.
ROOM_QUEUE_S = 20.0
ROOM_PLAYOUT_S = 15.0
# The source counts its speech played once the last frame is handed to
# WebRTC, with about 100-250 ms of it still on its way to the browser, which
# stops Otto's track when the relay leaves: the relay stays this much longer.
ROOM_PLAYOUT_TAIL_S = 0.5
# A registered room is in use while its call may still start (no call record
# yet, within this long of the mint: the start token's minute and the admit),
# while its call runs, and for this long after the call ended, which covers
# Otto's play-out. Every relay sweeps the registry this often and closes what
# is no longer in use; each sweep's heartbeat lasts this long, and without one
# the REST API starts no call.
ROOM_START_GRACE_S = 75.0
ROOM_END_GRACE_S = 30.0
ROOM_SWEEP_S = 30.0
ROOM_SWEEP_HEARTBEAT_S = 120
# A closed room's person has their tokens revoked again this long after the
# close, some sweeps before the first revocation lapses: a token LiveKit
# issued just before the close is taken for ROOM_TOKEN_LEEWAY_S past its
# expiry, past the first revocation. The room's entry is forgotten only once
# the second revocation and that leeway have both run out.
ROOM_REVOKE_AGAIN_AFTER_S = ROOM_REVOCATION_S - 4 * ROOM_SWEEP_S
ROOM_FORGET_AFTER_REVOKED_S = ROOM_REVOCATION_S + ROOM_TOKEN_LEEWAY_S
# Where the LiveKit project is this deployment's alone, a ``voice-`` room the
# registry has no entry for is adopted and closed once it is this old: longer
# than any call's room can live, so no live call ever loses its room to it.
ROOM_UNREGISTERED_GRACE_S = 600.0
# A sweep's listing of the project's rooms is given up past this.
ROOM_LIST_TIMEOUT_S = 10.0

# Why a call's room was closed: the call ended, or the sweep found it no
# longer in use (its call never started, or it outlived its call), open again
# after it was closed, or open with no entry in a project that is this
# deployment's alone.
RoomCloseWhy = Literal[
    "call_end", "never_started", "orphaned", "reopened", "unregistered"
]
# The person's audio reaches the call in frames this long: the relay cuts the
# room's audio into them, which the voice-activity check reads.
FRAME_S = 0.04

# What one call may spend, all meters together. Past it the call ends.
SPEND_CAP_USD = 0.50

# The scope classifier's model: cheap, and fast enough to label a turn while
# Otto is still answering it.
CLASSIFIER_MODEL = "gemini-3.8-flash"

# The transcript of record's model, and how many of a call's turns it
# transcribes at once. A turn is tried this many times on an error that may
# pass, each attempt given a few times the turn's length, and never less than
# the floor.
TRANSCRIBE_MODEL = "gemini-3.5-transcribe"
TRANSCRIBE_CONCURRENCY = 4
TRANSCRIBE_ATTEMPTS = 3
TRANSCRIBE_TIMEOUT_FLOOR_S = 10.0
TRANSCRIBE_TIMEOUT_PER_AUDIO_S = 2.0

# The recording is uploaded this often while a call runs, so a pod that dies
# loses at most this much of it.
CHECKPOINT_S = 30.0
# Every relay process looks for calls with no transcript that no process is
# transcribing when it starts and this often after, and runs at most this many
# jobs at once.
POSTCALL_SWEEP_S = 300.0
POSTCALL_MAX_JOBS = 4
# A transcript job, from its claim to the stored transcript, ends within this.
POSTCALL_JOB_TIMEOUT_S = 300.0
# A job's claim. The pod running a call claims its job as the call starts and
# gives the claim this long again with each refresh and as the job starts, so
# the claim outlives the call and its job, and a pod that dies leaves the job
# to another pod's look once it lapses.
POSTCALL_CLAIM_TTL_S = 900
# A call whose transcript job fails this many times records the failure.
POSTCALL_ATTEMPTS = 3
# How long a drain lets calls run on at most: ``VOICE_RELAY_DRAIN_MAX_S``'s
# default.
DEFAULT_DRAIN_S = 330.0


def transcribe_timeout_s(audio_s: float) -> float:
    """How long one attempt at transcribing *audio_s* seconds of a turn may
    take."""
    return TRANSCRIBE_TIMEOUT_FLOOR_S + TRANSCRIBE_TIMEOUT_PER_AUDIO_S * audio_s


# Audio tokens a second, for the share of a call the vendor bills but never
# reports in a usage message (the listening tail, a dropped session).
AUDIO_TOKENS_PER_S = 25


class Caps(BaseModel):
    """The limits of one call, sent to the browser in ``ready``."""

    model_config = ConfigDict(frozen=True)

    max_call_s: float = 300.0
    # Otto is nudged to sum up this long before the wall clock.
    soft_wrap_s: float = 30.0
    max_agent_turns: int = 16
    # Otto is nudged to sum up this many turns before the turn cap.
    soft_wrap_turns: int = 2
    # Seconds with nothing heard from the person, Otto's speech excluded.
    idle_s: float = 20.0

    def as_message(self) -> dict[str, float | int]:
        return self.model_dump()


CAPS = Caps()


class Timing(BaseModel):
    """How the relay paces a call."""

    model_config = ConfigDict(frozen=True)

    # The watchdog's tick.
    tick_s: float = 0.25
    # How often the pod checks whether someone asked to end the call.
    end_flag_poll_s: float = 1.0
    # A goodbye the relay asked for must finish within this, or it hangs up.
    wrap_grace_s: float = 12.0
    # After Otto's end tool, the goodbye must finish within this.
    finish_grace_s: float = 12.0
    # A dropped browser socket may re-attach within this.
    reattach_s: float = 15.0
    # What the call says to a detached browser is kept for its re-attach, up
    # to this much text; from the first message that does not fit, none is.
    outbox_bytes: int = 256 * 1024
    # After Otto's first audio, wait this long for the person's last caption
    # before their turn is labelled.
    scope_settle_s: float = 0.4
    scope_timeout_s: float = 8.0
    vendor_connect_timeout_s: float = 15.0
    max_vendor_reconnects: int = 2
    # One send or close on the browser's socket; past it the browser counts
    # as gone and the call holds for a re-attach.
    socket_timeout_s: float = 5.0
    # How often the person's recording is uploaded while the call runs.
    checkpoint_s: float = CHECKPOINT_S
    # The echo probe (``echo``) counts the microphone from this long after
    # Otto's first audio, while a working echo canceller settles.
    echo_probe_skip_s: float = 1.0
    # How long after Otto's audio his echo may still come back: the probe
    # counts until then, and a half-duplex call's microphone stays closed.
    echo_tail_s: float = 0.5


TIMING = Timing()


class Price(BaseModel):
    """A model's list price in USD per million tokens from a date on, and
    where it was read."""

    model_config = ConfigDict(frozen=True)

    source: str
    # The first day this rate applies (UTC).
    effective_from: date = date.min
    text_in: float = 0.0
    text_out: float = 0.0
    audio_in: float = 0.0
    audio_out: float = 0.0

    def per_token(self, meter: str) -> float:
        return getattr(self, meter) / 1_000_000


GEMINI_PRICING_SOURCE = "aistudio.google.com/docs/pricing, read 2026-11-03"

# Every model the relay pays for, each with its rates oldest first. A usage
# message is priced from here, so a model with no entry cannot run
# (``price_of`` raises at startup).
PRICES: dict[str, tuple[Price, ...]] = {
    "gemini-3.8-live": (
        Price(
            source=GEMINI_PRICING_SOURCE,
            text_in=0.75,
            audio_in=3.00,
            # Thinking tokens bill as text out.
            text_out=4.50,
            audio_out=12.00,
        ),
    ),
    CLASSIFIER_MODEL: (
        Price(source=GEMINI_PRICING_SOURCE, text_in=0.75, text_out=3.75),
        Price(
            source=GEMINI_PRICING_SOURCE + ", announced rate from 2027-01-01",
            effective_from=date(2027, 1, 1),
            text_in=1.50,
            text_out=7.50,
        ),
    ),
    # Its prompt is a clip of audio and one short line of text, all priced as
    # audio.
    TRANSCRIBE_MODEL: (
        Price(source=GEMINI_PRICING_SOURCE, audio_in=2.00, text_out=12.00),
    ),
}

# The last day the table is known to be right: its source lists rates through
# this date. Past it the relay goes on pricing calls at the latest rates it
# has and logs that the table needs reading again.
PRICES_VALID_UNTIL = date(2026, 12, 31)


def price_of(model: str, on: date | None = None) -> Price:
    """*model*'s rate on *on*, today (UTC) by default.

    Raises ``KeyError`` for a model with no price.
    """
    day = on or datetime.now(UTC).date()
    rates = PRICES[model]
    current = rates[0]
    for rate in rates:
        if rate.effective_from <= day:
            current = rate
    return current


def prices_out_of_date(on: date | None = None) -> bool:
    """Whether *on*, today (UTC) by default, is past the table's horizon."""
    return (on or datetime.now(UTC).date()) > PRICES_VALID_UNTIL
