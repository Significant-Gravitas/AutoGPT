"""What a voice call costs, metered as it happens.

Each vendor usage report becomes priced rows (Live audio in, text in, audio
out, and text out with thinking in it), each scope label two more, and each
turn the post-call job transcribes two more. Every row is one
``PlatformCostLog`` row through ``persist_and_record_usage`` on the
platform's account: the user's plan windows are left alone, since the
interview comes before any plan. Every row also adds to the platform's daily
voice spend in Redis, which the start route checks, and to Prometheus.

The vendor bills some of a call without ever reporting it: the listening
after its last usage report, or a session that dropped. That share is metered
too, as a labelled estimate from the audio the relay counted itself.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime

from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, ConfigDict

from backend.copilot.token_tracking import persist_and_record_usage
from backend.data.redis_client import AsyncRedisClient
from backend.voice_relay.config import (
    AUDIO_TOKENS_PER_S,
    CLASSIFIER_MODEL,
    IN_RATE,
    OUT_RATE,
    PRICES_VALID_UNTIL,
    SPEND_CAP_USD,
    TRANSCRIBE_MODEL,
    price_of,
    prices_out_of_date,
)
from backend.voice_relay.locks import add_daily_spend
from backend.voice_relay.vendor import Usage

logger = logging.getLogger(__name__)

BLOCK_NAME = "onboarding:voice_interview"
COST_SOURCE = "onboarding_voice_interview"
REDIS_TIMEOUT_S = 0.5

LIVE_CALLS = Gauge(
    "voice_interview_live_calls", "Onboarding voice interviews live on this relay"
)
CALLS = Counter(
    "voice_interview_calls_total",
    "Onboarding voice interviews, by how they ended and the leg they ran on",
    ["reason", "leg"],
)
CALL_SECONDS = Histogram(
    "voice_interview_call_seconds",
    "How long onboarding voice interviews ran",
    buckets=(15, 30, 60, 90, 120, 180, 240, 300, 330),
)
FIRST_AUDIO_SECONDS = Histogram(
    "voice_interview_first_audio_seconds",
    "From a call's start message to Otto's first audio, as the relay sees it",
    buckets=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
)
COST_USD = Counter(
    "voice_interview_cost_usd_total",
    "What onboarding voice interviews cost, by meter",
    ["meter"],
)
POSTCALL_SECONDS = Histogram(
    "voice_interview_postcall_seconds",
    "From a post-call job's start to the hand-off's answer",
    buckets=(1, 2, 3, 5, 8, 13, 20, 30, 60, 120),
)
POSTCALLS = Counter(
    "voice_interview_postcalls_total",
    "Post-call jobs, by how they ended: handed_off, failed, transcribed (waiting "
    "for the person's confirm), retry (left for another attempt), gave_up, or "
    "expired (the call's record was gone)",
    ["outcome"],
)

# The Prometheus role of each model a call pays for besides the live one.
_ROLES = {CLASSIFIER_MODEL: "classifier", TRANSCRIBE_MODEL: "transcribe"}


class Row(BaseModel):
    """One priced line of a call's bill."""

    model_config = ConfigDict(frozen=True)

    model: str
    meter: str
    tokens: int
    usd: float
    # Why, when the vendor never reported these tokens and the relay estimated them.
    estimate: str | None = None

    @property
    def is_input(self) -> bool:
        return self.meter.endswith("_in")

    @property
    def label(self) -> str:
        """The row's Prometheus meter: which model's which tokens."""
        return f"{_ROLES.get(self.model, 'live')}_{self.meter}"


def priced_rows(model: str, usage: Usage) -> list[Row]:
    """A usage report as priced rows; tokens it does not attribute are audio."""
    price = price_of(model)
    audio_in = usage.prompt_by_modality.get("audio", 0)
    text_in = usage.prompt_by_modality.get("text", 0)
    other_in = max(usage.prompt_tokens - audio_in - text_in, 0)
    audio_out = usage.response_by_modality.get("audio", 0)
    text_out = usage.response_by_modality.get("text", 0) + usage.thoughts_tokens
    other_out = max(usage.response_tokens - sum(usage.response_by_modality.values()), 0)
    quantities = (
        ("audio_in", audio_in + other_in),
        ("text_in", text_in),
        ("audio_out", audio_out + other_out),
        ("text_out", text_out),
    )
    return [
        Row(
            model=model,
            meter=meter,
            tokens=tokens,
            usd=tokens * price.per_token(meter),
        )
        for meter, tokens in quantities
        if tokens > 0
    ]


def estimated_rows(
    model: str,
    *,
    unreported_in_samples: int,
    unreported_out_samples: int,
    rebilled_prompt_tokens: int,
    why: str,
) -> list[Row]:
    """What the vendor billed but never reported, from the relay's own counts.

    *rebilled_prompt_tokens* is the context of a reply cut off mid-flight,
    which the vendor bills again without reporting it.
    """
    price = price_of(model)
    audio_in = (
        round(unreported_in_samples / IN_RATE * AUDIO_TOKENS_PER_S)
        + rebilled_prompt_tokens
    )
    audio_out = round(unreported_out_samples / OUT_RATE * AUDIO_TOKENS_PER_S)
    return [
        Row(
            model=model,
            meter=meter,
            tokens=tokens,
            usd=tokens * price.per_token(meter),
            estimate=why,
        )
        for meter, tokens in (("audio_in", audio_in), ("audio_out", audio_out))
        if tokens > 0
    ]


CostSink = Callable[[Row], Awaitable[None]]


class CallMeter:
    """One call's running bill."""

    def __init__(
        self,
        *,
        user_id: str,
        call_id: str,
        leg: str,
        model: str,
        redis: AsyncRedisClient | None,
        sink: CostSink | None = None,
    ) -> None:
        self.user_id = user_id
        self.call_id = call_id
        self.leg = leg
        self.model = model
        self.usd = 0.0
        self.rows: list[Row] = []
        self._redis = redis
        self._sink = sink or self._log_row
        self._pending: set[asyncio.Task] = set()
        # The audio the latest usage report covered, so an estimate counts
        # only what came after it.
        self._reported_in = 0
        self._reported_out = 0
        self._last_prompt_tokens = 0

    @property
    def over_cap(self) -> bool:
        return self.usd >= SPEND_CAP_USD

    async def live_usage(
        self, usage: Usage, *, sent_samples: int, received_samples: int
    ) -> None:
        """Meter a usage report of the live session.

        *sent_samples* and *received_samples* are the audio sent to the
        vendor and received from it so far, which this report covers.
        """
        self._reported_in, self._reported_out = sent_samples, received_samples
        self._last_prompt_tokens = usage.prompt_tokens
        await self._record(priced_rows(self.model, usage))

    async def classifier_usage(self, *, tokens_in: int, tokens_out: int) -> None:
        await self._model_usage(
            CLASSIFIER_MODEL, (("text_in", tokens_in), ("text_out", tokens_out))
        )

    async def transcription_usage(self, *, tokens_in: int, tokens_out: int) -> None:
        """Meter one turn's transcription: its audio in, its text out."""
        await self._model_usage(
            TRANSCRIBE_MODEL, (("audio_in", tokens_in), ("text_out", tokens_out))
        )

    async def _model_usage(
        self, model: str, quantities: tuple[tuple[str, int], ...]
    ) -> None:
        price = price_of(model)
        await self._record(
            [
                Row(
                    model=model,
                    meter=meter,
                    tokens=tokens,
                    usd=tokens * price.per_token(meter),
                )
                for meter, tokens in quantities
                if tokens > 0
            ]
        )

    async def unreported(
        self,
        *,
        sent_samples: int,
        received_samples: int,
        cut_off_reply: bool,
        why: str,
    ) -> None:
        """Meter what the live session billed since its last usage report.

        *cut_off_reply* says a reply was being generated when the session
        stopped, so its context is billed again.
        """
        rows = estimated_rows(
            self.model,
            unreported_in_samples=max(0, sent_samples - self._reported_in),
            unreported_out_samples=max(0, received_samples - self._reported_out),
            rebilled_prompt_tokens=self._last_prompt_tokens if cut_off_reply else 0,
            why=why,
        )
        self._reported_in, self._reported_out = sent_samples, received_samples
        await self._record(rows)

    async def flush(self) -> None:
        """Wait for the daily spend writes still in flight."""
        if self._pending:
            await asyncio.wait(list(self._pending), timeout=REDIS_TIMEOUT_S * 2)

    async def _record(self, rows: list[Row]) -> None:
        if not rows:
            return
        for row in rows:
            self.usd += row.usd
            self.rows.append(row)
            COST_USD.labels(meter=row.label).inc(row.usd)
            try:
                await self._sink(row)
            except Exception:
                logger.exception(
                    f"Voice call {self.call_id}: a cost row was not logged"
                )
        # Off the audio path: a slow Redis must not hold up the call.
        task = asyncio.create_task(self._add_daily(sum(row.usd for row in rows)))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _add_daily(self, usd: float) -> None:
        if self._redis is None:
            return
        try:
            await asyncio.wait_for(
                add_daily_spend(self._redis, usd, datetime.now(UTC)),
                timeout=REDIS_TIMEOUT_S,
            )
        except Exception:
            logger.warning(
                f"Voice call {self.call_id}: daily spend not counted", exc_info=True
            )

    async def _log_row(self, row: Row) -> None:
        metadata: dict[str, object] = {
            "call_id": self.call_id,
            "meter": row.meter,
            "leg": self.leg,
            "source": COST_SOURCE,
        }
        if row.estimate:
            metadata["estimate"] = row.estimate
        await persist_and_record_usage(
            session=None,
            user_id=self.user_id,
            prompt_tokens=row.tokens if row.is_input else 0,
            completion_tokens=0 if row.is_input else row.tokens,
            log_prefix="[VoiceInterview]",
            cost_usd=row.usd,
            model=row.model,
            provider="google",
            block_name_override=BLOCK_NAME,
            extra_metadata=metadata,
            charge_plan=False,
        )


DAY_S = 24 * 3600


def warn_if_prices_out_of_date(on: date | None = None) -> bool:
    """Log loudly when the price table is past its horizon. Never raises: the
    relay goes on pricing calls at the latest rates it has."""
    if not prices_out_of_date(on):
        return False
    logger.error(
        f"The voice relay's price table is past its horizon "
        f"({PRICES_VALID_UNTIL.isoformat()}): calls are priced at the latest "
        f"rates it has. Read the pricing page again and update PRICES in "
        f"backend/voice_relay/config.py."
    )
    return True


async def watch_prices(
    interval_s: float = DAY_S, today: Callable[[], date] | None = None
) -> None:
    """Repeat the price table check every *interval_s*, until cancelled."""
    while True:
        await asyncio.sleep(interval_s)
        warn_if_prices_out_of_date(today() if today else None)
