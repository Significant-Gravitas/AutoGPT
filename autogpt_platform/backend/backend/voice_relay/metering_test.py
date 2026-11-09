import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from prometheus_client import REGISTRY

from backend.copilot import token_tracking
from backend.voice_relay import locks, metering
from backend.voice_relay.config import IN_RATE, OUT_RATE, PRICES
from backend.voice_relay.metering import CallMeter, Row
from backend.voice_relay.vendor import Usage

LIVE = "gemini-3.8-live"
FLASH = "gemini-3.8-flash"


def _meter(redis, rows: list[Row] | None = None) -> CallMeter:
    async def sink(row: Row) -> None:
        if rows is not None:
            rows.append(row)

    return CallMeter(
        user_id="user-1",
        call_id="c" * 32,
        leg="websocket",
        model=LIVE,
        redis=redis,
        sink=sink,
    )


def test_a_usage_report_becomes_one_priced_row_per_meter():
    usage = Usage(
        prompt_tokens=900,
        response_tokens=100,
        thoughts_tokens=20,
        prompt_by_modality={"text": 700, "audio": 200},
        response_by_modality={"audio": 100},
    )
    rows = metering.priced_rows(LIVE, usage)
    price = PRICES[LIVE]
    assert [(row.meter, row.tokens) for row in rows] == [
        ("audio_in", 200),
        ("text_in", 700),
        ("audio_out", 100),
        # Thinking bills as text out.
        ("text_out", 20),
    ]
    assert rows[0].usd == pytest.approx(200 * price.audio_in / 1e6)
    assert rows[3].usd == pytest.approx(20 * price.text_out / 1e6)
    assert all(row.estimate is None for row in rows)


def test_tokens_a_report_does_not_attribute_are_priced_as_audio():
    usage = Usage(
        prompt_tokens=1000,
        response_tokens=300,
        prompt_by_modality={"text": 600},
        response_by_modality={"audio": 200},
    )
    rows = {row.meter: row.tokens for row in metering.priced_rows(LIVE, usage)}
    assert rows == {"audio_in": 400, "text_in": 600, "audio_out": 300}


def test_an_empty_report_has_no_rows():
    assert metering.priced_rows(LIVE, Usage()) == []


async def test_what_the_vendor_never_reported_is_metered_as_a_labelled_estimate(
    redis,
):
    rows: list[Row] = []
    meter = _meter(redis, rows)
    await meter.live_usage(
        Usage(prompt_tokens=500, prompt_by_modality={"text": 500}),
        sent_samples=IN_RATE * 10,
        received_samples=OUT_RATE * 4,
    )
    rows.clear()
    await meter.unreported(
        sent_samples=IN_RATE * 14,
        received_samples=OUT_RATE * 6,
        cut_off_reply=True,
        why="vendor session dropped",
    )
    by_meter = {row.meter: row for row in rows}
    # 4 s heard and 2 s spoken since the last report, at 25 tokens a second,
    # and the 500-token context billed again for the reply that was cut off.
    assert by_meter["audio_in"].tokens == 4 * 25 + 500
    assert by_meter["audio_out"].tokens == 2 * 25
    assert {row.estimate for row in rows} == {"vendor session dropped"}

    # The same audio is never estimated twice.
    rows.clear()
    await meter.unreported(
        sent_samples=IN_RATE * 14,
        received_samples=OUT_RATE * 6,
        cut_off_reply=False,
        why="listening after the last usage report",
    )
    assert rows == []


async def test_every_row_adds_to_the_platforms_daily_spend(redis):
    first, second = _meter(redis), _meter(redis)
    usage = Usage(prompt_tokens=10_000, prompt_by_modality={"audio": 10_000})
    await first.live_usage(usage, sent_samples=0, received_samples=0)
    await second.classifier_usage(tokens_in=1_000, tokens_out=100)
    await first.flush()
    await second.flush()
    now = datetime.now(UTC)
    spent = await locks.daily_spend_usd(redis, now)
    assert spent == pytest.approx(first.usd + second.usd)
    assert spent == pytest.approx(
        10_000 * PRICES[LIVE].audio_in / 1e6
        + 1_000 * PRICES[FLASH].text_in / 1e6
        + 100 * PRICES[FLASH].text_out / 1e6
    )
    key = locks.daily_spend_key(now)
    assert 0 < await redis.ttl(key) <= locks.DAILY_SPEND_TTL_S


async def test_a_call_is_over_its_cap_at_fifty_cents(redis):
    meter = _meter(redis)
    # 40,000 audio tokens out at $12 per million is $0.48.
    await meter.live_usage(
        Usage(response_tokens=40_000, response_by_modality={"audio": 40_000}),
        sent_samples=0,
        received_samples=0,
    )
    assert not meter.over_cap
    await meter.live_usage(
        Usage(response_tokens=2_000, response_by_modality={"audio": 2_000}),
        sent_samples=0,
        received_samples=0,
    )
    assert meter.over_cap


async def test_a_row_reaches_the_cost_log_on_the_platforms_account(monkeypatch, redis):
    logged: list = []
    record_cost = AsyncMock()
    monkeypatch.setattr(token_tracking, "_schedule_cost_log", logged.append)
    monkeypatch.setattr(token_tracking, "record_cost_usage", record_cost)
    meter = CallMeter(
        user_id="user-1", call_id="c" * 32, leg="websocket", model=LIVE, redis=redis
    )
    before = (
        REGISTRY.get_sample_value(
            "voice_interview_cost_usd_total", {"meter": "live_audio_out"}
        )
        or 0.0
    )

    await meter.live_usage(
        Usage(response_tokens=1_000, response_by_modality={"audio": 1_000}),
        sent_samples=0,
        received_samples=0,
    )
    await meter.unreported(
        sent_samples=IN_RATE, received_samples=0, cut_off_reply=False, why="tail"
    )
    await meter.flush()

    record_cost.assert_not_awaited()
    reported, estimated = logged
    assert reported.block_name == "onboarding:voice_interview"
    assert reported.provider == "google"
    assert reported.model == LIVE
    assert reported.output_tokens == 1_000 and reported.input_tokens == 0
    assert reported.cost_microdollars == 12_000
    assert reported.metadata["call_id"] == "c" * 32
    assert reported.metadata["meter"] == "audio_out"
    assert reported.metadata["leg"] == "websocket"
    assert reported.metadata["source"] == "onboarding_voice_interview"
    assert "estimate" not in reported.metadata
    assert estimated.metadata["estimate"] == "tail"
    assert estimated.input_tokens == 25
    after = REGISTRY.get_sample_value(
        "voice_interview_cost_usd_total", {"meter": "live_audio_out"}
    )
    assert after == pytest.approx(before + 0.012)


async def test_a_cost_log_failure_never_reaches_the_call(redis):
    async def broken(row: Row) -> None:
        raise RuntimeError("database is down")

    meter = CallMeter(
        user_id="user-1",
        call_id="c" * 32,
        leg="websocket",
        model=LIVE,
        redis=redis,
        sink=broken,
    )
    await meter.classifier_usage(tokens_in=100, tokens_out=10)
    assert meter.usd > 0


async def test_a_slow_redis_does_not_hold_up_a_usage_report():
    class SlowPipeline:
        def incrbyfloat(self, *args):
            pass

        def expire(self, *args):
            pass

        async def execute(self):
            await asyncio.sleep(5)

    class SlowRedis:
        def pipeline(self, transaction: bool = True):
            return SlowPipeline()

    meter = CallMeter(
        user_id="user-1",
        call_id="c" * 32,
        leg="websocket",
        model=LIVE,
        redis=SlowRedis(),  # type: ignore[arg-type]
        sink=AsyncMock(),
    )
    await asyncio.wait_for(
        meter.classifier_usage(tokens_in=100, tokens_out=10), timeout=0.05
    )
    await meter.flush()
