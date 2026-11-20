import asyncio
import logging
from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from prometheus_client import REGISTRY

from backend.copilot import token_tracking
from backend.voice_relay import locks, metering
from backend.voice_relay.config import (
    IN_RATE,
    OUT_RATE,
    PRICES,
    PRICES_VALID_UNTIL,
    TRANSCRIBE_MODEL,
    price_of,
)
from backend.voice_relay.conftest import until
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
    price = price_of(LIVE)
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
        10_000 * price_of(LIVE).audio_in / 1e6
        + 1_000 * price_of(FLASH).text_in / 1e6
        + 100 * price_of(FLASH).text_out / 1e6
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
    meter = CallMeter(user_id="user-1", call_id="c" * 32, model=LIVE, redis=redis)
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
    assert "leg" not in reported.metadata
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
        model=LIVE,
        redis=redis,
        sink=broken,
    )
    await meter.classifier_usage(tokens_in=100, tokens_out=10)
    assert meter.usd > 0


async def test_a_slow_redis_does_not_hold_up_a_usage_report(monkeypatch):
    # The daily-spend write may wait on Redis far longer than a report could.
    monkeypatch.setattr(metering, "REDIS_TIMEOUT_S", 60.0)
    answered = asyncio.Event()

    class SlowPipeline:
        def incrbyfloat(self, *args):
            pass

        def expire(self, *args):
            pass

        async def execute(self):
            await answered.wait()

    class SlowRedis:
        def pipeline(self, transaction: bool = True):
            return SlowPipeline()

    meter = CallMeter(
        user_id="user-1",
        call_id="c" * 32,
        model=LIVE,
        redis=SlowRedis(),  # type: ignore[arg-type]
        sink=AsyncMock(),
    )
    await asyncio.wait_for(
        meter.classifier_usage(tokens_in=100, tokens_out=10), timeout=5
    )
    # The report is in while its daily-spend write still waits on Redis.
    assert meter.usd > 0
    assert any(not task.done() for task in meter._pending)
    answered.set()
    await meter.flush()


async def test_a_transcribed_turn_is_billed_as_audio_in_and_text_out(redis):
    rows: list[Row] = []
    meter = _meter(redis, rows)

    await meter.transcription_usage(tokens_in=1_000, tokens_out=50)
    await meter.flush()

    price = price_of(TRANSCRIBE_MODEL)
    assert [(row.model, row.meter, row.tokens) for row in rows] == [
        (TRANSCRIBE_MODEL, "audio_in", 1_000),
        (TRANSCRIBE_MODEL, "text_out", 50),
    ]
    assert meter.usd == pytest.approx(
        1_000 * price.audio_in / 1e6 + 50 * price.text_out / 1e6
    )
    assert [row.label for row in rows] == ["transcribe_audio_in", "transcribe_text_out"]
    assert await locks.daily_spend_usd(redis, datetime.now(UTC)) == pytest.approx(
        meter.usd
    )


def test_the_classifiers_rate_doubles_on_its_announced_date():
    before = price_of(FLASH, date(2026, 12, 31))
    after = price_of(FLASH, date(2027, 1, 1))
    assert (before.text_in, before.text_out) == (0.75, 3.75)
    assert (after.text_in, after.text_out) == (1.50, 7.50)
    assert after.effective_from == date(2027, 1, 1)


def test_every_models_rates_run_oldest_first():
    for model, rates in PRICES.items():
        days = [rate.effective_from for rate in rates]
        assert days == sorted(set(days)), model


def test_a_price_table_past_its_horizon_is_logged_loudly_and_never_refuses(caplog):
    with caplog.at_level(logging.ERROR, logger=metering.__name__):
        assert not metering.warn_if_prices_out_of_date(PRICES_VALID_UNTIL)
        assert caplog.records == []
        day_after = PRICES_VALID_UNTIL + timedelta(days=1)
        assert metering.warn_if_prices_out_of_date(day_after)
    assert [record.levelno for record in caplog.records] == [logging.ERROR]
    assert "past its horizon" in caplog.records[0].getMessage()


async def test_the_price_table_is_checked_again_while_the_relay_runs(caplog):
    day_after = PRICES_VALID_UNTIL + timedelta(days=1)
    with caplog.at_level(logging.ERROR, logger=metering.__name__):
        watch = asyncio.create_task(
            metering.watch_prices(interval_s=0.01, today=lambda: day_after)
        )
        try:
            await until(lambda: len(caplog.records) >= 3)
        finally:
            watch.cancel()


def test_thinking_is_priced_once_as_text_out():
    # A turn as Gemini Live reports it: the response is the audio Otto speaks,
    # thinking is counted apart from it, and the total is prompt plus response.
    usage = Usage(
        prompt_tokens=1084,
        response_tokens=224,
        thoughts_tokens=63,
        prompt_by_modality={"text": 774, "audio": 205},
        response_by_modality={"audio": 224},
    )
    rows = {row.meter: row.tokens for row in metering.priced_rows(LIVE, usage)}
    assert rows == {"audio_in": 310, "text_in": 774, "audio_out": 224, "text_out": 63}


def test_a_report_without_a_response_breakdown_prices_thinking_once():
    usage = Usage(prompt_tokens=500, response_tokens=300, thoughts_tokens=50)
    rows = {row.meter: row.tokens for row in metering.priced_rows(LIVE, usage)}
    assert rows == {"audio_in": 500, "audio_out": 300, "text_out": 50}
