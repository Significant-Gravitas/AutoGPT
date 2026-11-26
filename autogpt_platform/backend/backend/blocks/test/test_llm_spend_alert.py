"""The spend alert times a cut-off LLM call on its last HTTP request (#16173).

One timeout covers every request the provider SDK sends for a block call, its
own retries included, and the backoff between them. Only the request the
timeout cut off can have been generating, so the alert reads how long that one
had been open, and finds nothing open when the cut-off landed in a backoff.

These run the real ``llm_call`` on a fake clock (``llm._now``) that only the
test moves, with a fake SDK that records the requests it sends.
``test_llm_spend_alert_sdk.py`` checks the same rules through each real SDK.
"""

import logging
from typing import Any

import pytest

import backend.blocks.llm as llm
from backend.blocks.test import llm_spend_alert_fixtures
from backend.blocks.test.llm_spend_alert_fixtures import (
    PROMPT,
    Clock,
    alert_fields,
    timeout_warning,
)
from backend.util.llm.http_attempts import current_http_attempts

# Bound here so pytest finds the shared fixtures in this module.
clock = llm_spend_alert_fixtures.clock
logs = llm_spend_alert_fixtures.logs

# A request the SDK sent ``seconds`` into the call, and the status it got
# back, or None for no response.
Step = tuple[float, int | None]


async def _cut_off(
    monkeypatch: pytest.MonkeyPatch,
    clock: Clock,
    slice_: float,
    steps: list[Step],
    cut_at: float | None = None,
) -> None:
    """``llm_call`` with an SDK that sends ``steps``, then the deadline cutting
    it off at ``cut_at``, by default the end of the slice."""

    async def sdk_with_retries(**kwargs: Any) -> llm.LLMResponse:
        log = current_http_attempts.get()
        assert log is not None, "llm_call must set the attempt log"
        for seconds, status in steps:
            clock.at(seconds)
            log.request_sent()
            if status is not None:
                log.response_received(status)
        clock.at(slice_ if cut_at is None else cut_at)
        raise TimeoutError()

    monkeypatch.setattr(llm, "_llm_call", sdk_with_retries)
    with pytest.raises(TimeoutError):
        await llm.llm_call(
            credentials=llm.TEST_CREDENTIALS,
            llm_model=llm.DEFAULT_LLM_MODEL,
            prompt=PROMPT,
            max_tokens=100,
            on_abandoned=None,
            timeout_seconds=slice_,
        )


@pytest.mark.parametrize(
    "slice_, steps, in_flight",
    [
        pytest.param(30, [(0, None)], 30.0, id="hang"),
        # The headers came back; the body was still arriving.
        pytest.param(100, [(0, 200)], 100.0, id="200-whose-body-was-cut-off"),
        pytest.param(600, [(0, 500), (1.5, None)], 598.5, id="fast-500-long-retry"),
        pytest.param(600, [(0, 500), (560, None)], 40.0, id="retry-open-over-30s"),
    ],
)
@pytest.mark.asyncio
async def test_a_long_open_last_request_alerts(
    monkeypatch, clock, logs, slice_, steps, in_flight
):
    await _cut_off(monkeypatch, clock, slice_, steps)

    [fields] = alert_fields(logs)
    assert fields["configured_timeout_seconds"] == slice_
    assert fields["last_request_in_flight_seconds"] == in_flight
    assert fields["http_attempts"] == len(steps)
    assert fields["response_statuses"] == [status for _, status in steps]
    assert fields["in_flight_source"] == "last_request"


@pytest.mark.parametrize(
    "slice_, steps, in_flight_source",
    [
        # #16173: a slow error, then the deadline cut the retry off early.
        pytest.param(
            30, [(0, 500), (24.5, None)], "last_request", id="slow-500-then-cut-retry"
        ),
        pytest.param(
            600, [(0, 500), (580, None)], "last_request", id="retry-open-under-30s"
        ),
        # The deadline landed in the SDK's backoff: the last request already
        # had its error response, so nothing was open.
        pytest.param(30, [(0, 500)], "retry_backoff", id="backoff-after-a-500"),
        pytest.param(600, [(0, 429)], "retry_backoff", id="backoff-after-a-429"),
        pytest.param(
            600,
            [(0, 500), (2, 503)],
            "retry_backoff",
            id="backoff-after-a-second-error",
        ),
        # A request that got no response keeps its place, so the 503 is read
        # as the answer to the last request, not to the dropped one.
        pytest.param(
            600,
            [(0, None), (2, 503)],
            "retry_backoff",
            id="backoff-after-a-dropped-request",
        ),
    ],
)
@pytest.mark.asyncio
async def test_no_request_open_long_at_the_cut_off_does_not_alert(
    monkeypatch, clock, logs, slice_, steps, in_flight_source
):
    await _cut_off(monkeypatch, clock, slice_, steps)

    assert alert_fields(logs) == []
    warning = timeout_warning(logs)
    statuses = [status for _, status in steps]
    assert f"{len(steps)} HTTP attempt(s) (statuses {statuses})" in warning
    assert f"({in_flight_source})" in warning


@pytest.mark.parametrize(
    "cut_at, alerts", [pytest.param(95, True), pytest.param(50, False)]
)
@pytest.mark.asyncio
async def test_without_recorded_requests_total_elapsed_decides(
    monkeypatch, clock, logs, cut_at, alerts
):
    """A provider path without the hooks keeps the whole-call rule."""
    await _cut_off(monkeypatch, clock, 100, [], cut_at=cut_at)

    found = alert_fields(logs)
    assert bool(found) is alerts
    if alerts:
        assert found[0]["in_flight_source"] == "elapsed_fallback"
        assert found[0]["last_request_in_flight_seconds"] == 95.0


@pytest.mark.asyncio
async def test_a_fast_answer_logs_no_timeout(monkeypatch, clock, logs):
    async def answers(**kwargs: Any) -> llm.LLMResponse:
        log = current_http_attempts.get()
        assert log is not None
        log.request_sent()
        log.response_received(200)
        return llm.LLMResponse(
            raw_response="",
            prompt=[],
            response="ok",
            tool_calls=None,
            prompt_tokens=1,
            completion_tokens=1,
        )

    monkeypatch.setattr(llm, "_llm_call", answers)
    result = await llm.llm_call(
        credentials=llm.TEST_CREDENTIALS,
        llm_model=llm.DEFAULT_LLM_MODEL,
        prompt=PROMPT,
        max_tokens=100,
        on_abandoned=None,
    )

    assert result.response == "ok"
    assert [r for r in logs if r.levelno >= logging.WARNING] == []
    assert current_http_attempts.get() is None
