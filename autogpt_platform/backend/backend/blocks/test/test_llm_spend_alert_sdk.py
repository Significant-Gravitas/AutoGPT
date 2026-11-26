"""The spend-alert rules of ``test_llm_spend_alert.py`` (#16173), end to end
through each real provider SDK.

The SDK's own retry loop sends the requests, the hooked httpx client records
them, and the deadline cancels the SDK mid-request or mid-backoff. Each SDK's
slow 500 or 504, and Anthropic's 529, is a case the false alert hit. The
``llm`` module's own ``asyncio`` name is swapped for one whose ``wait_for``
gives up exactly where the test puts the deadline, on the fake clock.
"""

import asyncio
import contextlib
from types import ModuleType
from typing import Any

import anthropic
import groq
import httpx
import openai
import pytest

import backend.blocks.llm as llm
from backend.blocks.test import llm_spend_alert_fixtures
from backend.blocks.test.llm_spend_alert_fixtures import (
    PROMPT,
    Clock,
    alert_fields,
    timeout_warning,
)

# Bound here so pytest finds the shared fixtures in this module.
clock = llm_spend_alert_fixtures.clock
logs = llm_spend_alert_fixtures.logs

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
    clock: Clock,
    sdk_name: str,
    replies: list[tuple[float, int]],
    in_backoff: bool = False,
    close_takes: float = 0,
) -> None:
    """``llm_call`` through the real SDK. ``replies`` are (seconds, status) for
    the requests it answers; the request after them hangs until the deadline
    cancels it. With ``in_backoff`` the deadline lands instead while the SDK
    waits out the last reply's retry-after. Closing the HTTP client takes
    ``close_takes`` seconds."""
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
        client = real_factory(**kwargs, transport=httpx.MockTransport(provider))
        real_aclose = client.aclose

        async def slow_aclose() -> None:
            clock.now += close_takes
            await real_aclose()

        monkeypatch.setattr(client, "aclose", slow_aclose)
        clients.append(client)
        return client

    monkeypatch.setattr(sdk, "DefaultAsyncHttpxClient", hooked_factory)
    monkeypatch.setattr(llm, "asyncio", _DeadlineWhenCut(cut))
    with pytest.raises(TimeoutError):
        await llm.llm_call(
            credentials=llm.TEST_CREDENTIALS,
            llm_model=model,
            prompt=PROMPT,
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

    assert alert_fields(logs) == []
    warning = timeout_warning(logs)
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

    assert alert_fields(logs) == []
    warning = timeout_warning(logs)
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

    [fields] = alert_fields(logs)
    assert fields["provider"] == sdk_name
    assert fields["last_request_in_flight_seconds"] == in_flight
    assert fields["http_attempts"] == len(replies) + 1
    assert fields["response_statuses"] == statuses
    assert fields["in_flight_source"] == "last_request"


@pytest.mark.parametrize("sdk_name", list(_SDKS))
@pytest.mark.asyncio
async def test_each_sdk_closing_its_client_is_not_time_in_flight(
    monkeypatch, clock, logs, sdk_name
):
    """``wait_for`` raises only after the cancelled SDK call has closed its
    HTTP client. A retry open 25 s at the cut-off stays under the 27 s alert
    threshold however long that close takes."""
    await _cut_off_through_sdk(monkeypatch, clock, sdk_name, [(5, 500)], close_takes=10)

    assert alert_fields(logs) == []
    warning = timeout_warning(logs)
    assert "timed out after 30.0s and 2 HTTP attempt(s)" in warning
    assert "final request open 25.0s (last_request)" in warning
