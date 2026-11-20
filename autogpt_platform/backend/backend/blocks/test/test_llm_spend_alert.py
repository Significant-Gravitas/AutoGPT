"""The spend alert times a cut-off LLM call on its last HTTP request (#16173).

One timeout covers every request the provider SDK sends for a block call, its
own retries included, and the backoff between them. Only the request the
timeout cut off can have been generating, so the alert reads how long that one
had been open, and finds nothing open when the cut-off landed in a backoff.

These run the real ``llm_call`` on a fake clock (``llm._now``) that only the
test moves. The SDK tests swap the ``llm`` module's own ``asyncio`` name for
one whose ``wait_for`` gives up exactly where the test puts the deadline.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import anthropic
import groq
import httpx
import openai
import pytest

import backend.blocks.llm as llm
from backend.util.llm.http_attempts import current_http_attempts

_BUDGET = 600
_PROMPT = [{"role": "user", "content": "Hello"}]

# A request the SDK sent ``seconds`` into the call, and the status it got
# back, or None for no response.
Step = tuple[float, int | None]


class _Clock:
    def __init__(self) -> None:
        # A real base keeps the saturation tracker's ages sane.
        self.start = float(int(time.monotonic()))
        self.now = self.start

    def at(self, seconds: float) -> None:
        self.now = self.start + seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(llm, "_now", lambda: clock.now)
    monkeypatch.setattr(llm, "LLM_REQUEST_TIMEOUT_SECONDS", _BUDGET)
    return clock


@pytest.fixture
def logs() -> Iterator[list[logging.LogRecord]]:
    """What the llm module logs during the test. The handler sits on the
    module's own logger: caplog's root handler sees nothing under the app's
    logging config."""
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    target = logging.getLogger(llm.__name__)
    handler = _Collect(level=logging.DEBUG)
    previous = target.level
    target.addHandler(handler)
    target.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        target.removeHandler(handler)
        target.setLevel(previous)


def _alert_fields(records: list[logging.LogRecord]) -> list[dict[str, Any]]:
    # Containment: the app's log config can wrap the message in colour codes.
    return [
        vars(r)["json_fields"]
        for r in records
        if r.levelno == logging.ERROR and llm.SPEND_BURNED_ALERT in r.getMessage()
    ]


def _timeout_warning(records: list[logging.LogRecord]) -> str:
    warnings = [
        r.getMessage()
        for r in records
        if r.levelno == logging.WARNING and "LLM call timed out" in r.getMessage()
    ]
    assert len(warnings) == 1, warnings
    return warnings[0]


async def _cut_off(
    monkeypatch: pytest.MonkeyPatch,
    clock: _Clock,
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
            prompt=_PROMPT,
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

    [fields] = _alert_fields(logs)
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

    assert _alert_fields(logs) == []
    warning = _timeout_warning(logs)
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

    found = _alert_fields(logs)
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
        prompt=_PROMPT,
        max_tokens=100,
        on_abandoned=None,
    )

    assert result.response == "ok"
    assert [r for r in logs if r.levelno >= logging.WARNING] == []
    assert current_http_attempts.get() is None


# The same rules end to end through each real SDK: its own retry loop sends the
# requests, the hooked httpx client records them, and the deadline cancels the
# SDK mid-request or mid-backoff. Diag reproduced the false alert on v0.9.0 for
# every SDK with a slow 500 or 504, and on Anthropic's 529.
_SDKS: dict[str, tuple[ModuleType, llm.LLMModel]] = {
    "openai": (openai, llm.DEFAULT_LLM_MODEL),
    "anthropic": (anthropic, llm.LLMModel.CLAUDE_4_6_SONNET),
    "groq": (groq, llm.LLMModel.LLAMA3_3_70B),
}
_SDK_SLICE = 30


class _DeadlineWhenCut:
    """The llm module's ``asyncio``, whose ``wait_for`` gives up once ``cut``
    is set, as the real one would at the deadline, with no real waiting."""

    def __init__(self, cut: asyncio.Event) -> None:
        self._cut = cut

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)

    async def wait_for(self, aw: Any, timeout: float) -> Any:
        assert timeout == _SDK_SLICE
        call = asyncio.ensure_future(aw)
        cut = asyncio.ensure_future(self._cut.wait())
        await asyncio.wait({call, cut}, return_when=asyncio.FIRST_COMPLETED)
        cut.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cut
        if call.done():
            return call.result()
        call.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await call
        raise asyncio.TimeoutError()


async def _cut_off_through_sdk(
    monkeypatch: pytest.MonkeyPatch,
    clock: _Clock,
    sdk_name: str,
    replies: list[tuple[float, int]],
    in_backoff: bool = False,
) -> None:
    """``llm_call`` through the real SDK. ``replies`` are (seconds, status) for
    the requests it answers; the request after them hangs until the deadline
    cancels it. With ``in_backoff`` the deadline lands instead while the SDK
    waits out the last reply's retry-after."""
    sdk, model = _SDKS[sdk_name]
    cut = asyncio.Event()
    pending = list(replies)

    async def provider(request: httpx.Request) -> httpx.Response:
        if not pending:
            clock.at(_SDK_SLICE)
            cut.set()
            await asyncio.Event().wait()  # until the deadline cancels it
        seconds, status = pending.pop(0)
        clock.now += seconds
        retry_after_ms = "1"  # keeps the SDK's backoff out of the test's time
        if in_backoff and not pending:
            clock.at(_SDK_SLICE)
            cut.set()
            retry_after_ms = "20000"
        return httpx.Response(
            status,
            json={"error": {"type": "api_error", "message": "upstream"}},
            headers={"retry-after-ms": retry_after_ms},
        )

    clients: list[httpx.AsyncClient] = []
    real_factory = sdk.DefaultAsyncHttpxClient

    def hooked_factory(**kwargs: Any) -> httpx.AsyncClient:
        clients.append(real_factory(**kwargs, transport=httpx.MockTransport(provider)))
        return clients[-1]

    monkeypatch.setattr(sdk, "DefaultAsyncHttpxClient", hooked_factory)
    monkeypatch.setattr(llm, "asyncio", _DeadlineWhenCut(cut))
    with pytest.raises(TimeoutError):
        await llm.llm_call(
            credentials=llm.TEST_CREDENTIALS,
            llm_model=model,
            prompt=_PROMPT,
            max_tokens=100,
            on_abandoned=None,
            timeout_seconds=_SDK_SLICE,
        )
    assert cut.is_set() and not pending, "the SDK never sent what the test expects"
    assert len(clients) == 1 and clients[0].is_closed


@pytest.mark.parametrize(
    "sdk_name, status",
    [(name, status) for name in _SDKS for status in (500, 504)] + [("anthropic", 529)],
)
@pytest.mark.asyncio
async def test_each_sdk_slow_error_then_cut_off_retry_does_not_alert(
    monkeypatch, clock, logs, sdk_name, status
):
    await _cut_off_through_sdk(monkeypatch, clock, sdk_name, [(24.5, status)])

    assert _alert_fields(logs) == []
    warning = _timeout_warning(logs)
    assert f"2 HTTP attempt(s) (statuses [{status}, None])" in warning
    assert "final request open 5.5s (last_request)" in warning


@pytest.mark.parametrize(
    "sdk_name, status", [(name, status) for name in _SDKS for status in (429, 503)]
)
@pytest.mark.asyncio
async def test_each_sdk_cut_off_in_its_retry_backoff_does_not_alert(
    monkeypatch, clock, logs, sdk_name, status
):
    await _cut_off_through_sdk(
        monkeypatch, clock, sdk_name, [(24.5, status)], in_backoff=True
    )

    assert _alert_fields(logs) == []
    warning = _timeout_warning(logs)
    assert f"1 HTTP attempt(s) (statuses [{status}])" in warning
    assert "final request open 0.0s (retry_backoff)" in warning


@pytest.mark.parametrize("sdk_name", list(_SDKS))
@pytest.mark.parametrize(
    "replies, in_flight, statuses",
    [
        pytest.param([], 30.0, [None], id="hang"),
        pytest.param([(0.5, 500)], 29.5, [500, None], id="fast-500-long-retry"),
    ],
)
@pytest.mark.asyncio
async def test_each_sdk_long_open_request_still_alerts(
    monkeypatch, clock, logs, sdk_name, replies, in_flight, statuses
):
    await _cut_off_through_sdk(monkeypatch, clock, sdk_name, replies)

    [fields] = _alert_fields(logs)
    assert fields["provider"] == sdk_name
    assert fields["last_request_in_flight_seconds"] == in_flight
    assert fields["http_attempts"] == len(replies) + 1
    assert fields["response_statuses"] == statuses
    assert fields["in_flight_source"] == "last_request"
