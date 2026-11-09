"""The voice relay's limits, prices and wire constants.

The caps are the product's limits for an interview before the paywall. The
relay enforces them and the browser only displays them; nothing a caller sends
can change them, and a test that needs shorter ones builds its own.
"""

from pydantic import BaseModel, ConfigDict

# The browser's microphone and Gemini Live's speech, both mono PCM16.
IN_RATE = 16_000
OUT_RATE = 24_000

# An audio frame from the browser past this size is dropped, not buffered.
MAX_FRAME_BYTES = 16_000
# Control messages from the browser are JSON text frames under this size.
MAX_TEXT_BYTES = 4_096
# The server's own frame limit, above MAX_FRAME_BYTES so an oversized frame
# reaches the relay, which drops it, instead of closing the call.
WS_MAX_MESSAGE_BYTES = 64 * 1024
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

# What one call may spend, all meters together. Past it the call ends.
SPEND_CAP_USD = 0.50

# The scope classifier's model: cheap, and fast enough to label a turn while
# Otto is still answering it.
CLASSIFIER_MODEL = "gemini-3.8-flash"

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
    # Otto's audio kept for a detached browser, in seconds of speech.
    outbox_s: float = 20.0
    # After Otto's first audio, wait this long for the person's last caption
    # before their turn is labelled.
    scope_settle_s: float = 0.4
    scope_timeout_s: float = 8.0
    vendor_connect_timeout_s: float = 15.0
    max_vendor_reconnects: int = 2


TIMING = Timing()


class Price(BaseModel):
    """A model's list price in USD per million tokens, and where it was read."""

    model_config = ConfigDict(frozen=True)

    source: str
    text_in: float = 0.0
    text_out: float = 0.0
    audio_in: float = 0.0
    audio_out: float = 0.0

    def per_token(self, meter: str) -> float:
        return getattr(self, meter) / 1_000_000


GEMINI_PRICING_SOURCE = (
    "aistudio.google.com/docs/pricing, read 2026-11-03 (rates through 2026-12-31)"
)

# Every model the relay pays for. A usage message is priced from here, so a
# model with no entry cannot run (``price_of`` raises at startup).
PRICES: dict[str, Price] = {
    "gemini-3.8-live": Price(
        source=GEMINI_PRICING_SOURCE,
        text_in=0.75,
        audio_in=3.00,
        # Thinking tokens bill as text out.
        text_out=4.50,
        audio_out=12.00,
    ),
    CLASSIFIER_MODEL: Price(
        source=GEMINI_PRICING_SOURCE + "; doubles on 2027-01-01",
        text_in=0.75,
        text_out=3.75,
    ),
}


def price_of(model: str) -> Price:
    """The price of *model*. Raises ``KeyError`` for a model with no price."""
    return PRICES[model]
