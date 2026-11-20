"""An LLM block's retries share one deadline (#14293).

Each attempt used to get a fresh ``LLM_REQUEST_TIMEOUT_SECONDS``, so slow
failures that are not timeouts ran every retry at full length. These tests run
the real retry loop on a fake clock: each fake provider call advances the clock
instead of sleeping, and a call as long as its slice times out at the slice, as
the real ``wait_for`` would.
"""

import asyncio
import logging
import time
from collections.abc import Iterator
from typing import cast

import pytest

import backend.blocks.llm as llm

_CREDENTIALS = cast(llm.AICredentials, llm.TEST_CREDENTIALS_INPUT)
_TAG = "test123456"
_MISSING_KEY = f'<json_output id="{_TAG}">{{"wrong": "key"}}</json_output>'
_VALID = f'<json_output id="{_TAG}">{{"key": "value"}}</json_output>'
_BUDGET = 600

Step = tuple[float, str | BaseException]


def _deadline_block(
    monkeypatch: pytest.MonkeyPatch, steps: list[Step], budget: int = _BUDGET
) -> tuple[llm.AIStructuredResponseGeneratorBlock, dict[str, float], list[float]]:
    """A block whose provider calls run ``steps`` in order on a fake clock.

    A step is ``(seconds, outcome)``: the call takes ``seconds``, then returns
    ``outcome`` as the answer or raises it. Returns the block, the clock and
    the slice each attempt got.
    """
    monkeypatch.setattr(llm, "LLM_REQUEST_TIMEOUT_SECONDS", budget)
    clock = {"t": float(int(time.monotonic()))}  # real base: sane saturation ages
    monkeypatch.setattr(llm, "_now", lambda: clock["t"])
    slices: list[float] = []
    pending = list(steps)

    async def fake_provider_call(**kwargs) -> llm.LLMResponse:
        seconds, outcome = pending.pop(0)
        if seconds >= slices[-1]:
            clock["t"] += slices[-1]
            raise TimeoutError()
        clock["t"] += seconds
        if isinstance(outcome, BaseException):
            raise outcome
        return llm.LLMResponse(
            raw_response="",
            prompt=[],
            response=outcome,
            tool_calls=None,
            prompt_tokens=1,
            completion_tokens=1,
            provider_cost=0.25,
        )

    monkeypatch.setattr(llm, "_llm_call", fake_provider_call)
    block = llm.AIStructuredResponseGeneratorBlock()
    monkeypatch.setattr(
        block, "get_collision_proof_output_tag_id", lambda expected_format: _TAG
    )
    real_llm_call = block.llm_call

    async def recording_llm_call(**kwargs) -> llm.LLMResponse:
        slices.append(kwargs.get("timeout_seconds", budget))
        return await real_llm_call(**kwargs)

    monkeypatch.setattr(block, "llm_call", recording_llm_call)
    return block, clock, slices


async def _run(block: llm.AIStructuredResponseGeneratorBlock) -> dict:
    input_data = llm.AIStructuredResponseGeneratorBlock.Input(
        prompt="Hello",
        expected_format={"key": "value"},
        model=llm.DEFAULT_LLM_MODEL,
        credentials=_CREDENTIALS,
        retry=3,
    )
    return {
        name: value
        async for name, value in block.run(input_data, credentials=llm.TEST_CREDENTIALS)
    }


@pytest.fixture
def spend_alerts() -> Iterator[list[logging.LogRecord]]:
    """The spend alerts the llm module logs during the test. The handler sits
    on the module's own logger: caplog's root handler sees nothing under the
    app's logging config."""
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if llm.SPEND_BURNED_ALERT in record.getMessage():
                records.append(record)

    target = logging.getLogger(llm.__name__)
    handler = _Collect(level=logging.DEBUG)
    target.addHandler(handler)
    try:
        yield records
    finally:
        target.removeHandler(handler)


def test_remaining_attempt_timeout_uses_leftover_budget():
    per_attempt = 10.0
    deadline = 100.0
    assert llm.remaining_attempt_timeout(deadline, per_attempt, now=90.0) == 10.0
    assert llm.remaining_attempt_timeout(deadline, per_attempt, now=94.0) == 6.0
    assert llm.remaining_attempt_timeout(deadline, per_attempt, now=100.0) == 0.0


@pytest.mark.parametrize(
    "budget, first_attempt",
    [
        pytest.param(_BUDGET, 120, id="default-budget"),
        pytest.param(_BUDGET, 569, id="default-budget-31s-left"),
        # The shortest budget the settings allow: its floor is half the budget,
        # so a fast failure is still retried.
        pytest.param(30, 1, id="shortest-budget"),
        pytest.param(30, 14, id="shortest-budget-16s-left"),
    ],
)
@pytest.mark.asyncio
async def test_structured_block_retries_use_remaining_deadline(
    monkeypatch, budget, first_attempt
):
    """A slow-then-malformed first attempt must not get a fresh full timeout:
    the retry gets what is left of the shared deadline."""
    block, _, slices = _deadline_block(
        monkeypatch, [(first_attempt, _MISSING_KEY), (1, _VALID)], budget
    )

    outputs = await _run(block)

    assert outputs["response"] == {"key": "value"}
    assert slices == [budget, budget - first_attempt]


@pytest.mark.asyncio
async def test_slow_failures_stay_within_the_budget(monkeypatch):
    """Slow non-timeout failures used to get a fresh full timeout each, so
    three of them ran past the budget (#14293)."""
    slow_500 = RuntimeError("upstream 500")
    block, clock, slices = _deadline_block(monkeypatch, [(250, slow_500)] * 3)
    started = clock["t"]

    with pytest.raises(RuntimeError):
        await _run(block)

    assert clock["t"] - started <= _BUDGET
    assert slices == [_BUDGET, 350, 100]


@pytest.mark.asyncio
async def test_timeout_on_a_shortened_retry_still_alerts(monkeypatch, spend_alerts):
    """The spend alert compares elapsed time with the slice the attempt got,
    not the whole budget, or a cut-off final retry reads as a fast fail."""
    block, _, slices = _deadline_block(
        monkeypatch, [(500, _MISSING_KEY), (1000, "never returned")]
    )

    with pytest.raises(RuntimeError) as err:
        await _run(block)

    assert slices == [_BUDGET, 100]
    assert "exceeded 100.0s" in str(err.value)
    assert len(spend_alerts) == 1
    assert spend_alerts[0].levelno == logging.ERROR
    assert vars(spend_alerts[0])["json_fields"]["configured_timeout_seconds"] == 100


@pytest.mark.asyncio
async def test_a_cut_off_retry_shows_its_slice_rounded(monkeypatch, spend_alerts):
    """What is left of the deadline on the real clock is never round. The
    message rounds the retry's slice; the alert keeps the exact number."""
    block, _, slices = _deadline_block(
        monkeypatch, [(512.56780124, _MISSING_KEY), (1000, "never returned")]
    )

    with pytest.raises(RuntimeError) as err:
        await _run(block)

    assert slices[1] == pytest.approx(87.43219876)
    assert "exceeded 87.4s and was cancelled" in str(err.value)
    [alert] = spend_alerts
    assert vars(alert)["json_fields"]["configured_timeout_seconds"] == slices[1]


@pytest.mark.parametrize(
    "budget, first_attempt",
    [
        pytest.param(_BUDGET, 571, id="default-budget-29s-left"),
        pytest.param(30, 16, id="shortest-budget-14s-left"),
    ],
)
@pytest.mark.asyncio
async def test_floor_stops_retrying_and_says_the_budget_ran_out(
    monkeypatch, budget, first_attempt
):
    """A retry left with less than the floor would only be cut off, so the
    loop stops, names the budget, keeps the last attempt's error, and still
    records what was spent."""
    block, _, slices = _deadline_block(
        monkeypatch, [(first_attempt, _MISSING_KEY)], budget
    )

    with pytest.raises(RuntimeError) as err:
        await _run(block)

    assert slices == [budget]
    assert str(err.value).startswith(
        f"LLM retry budget of {budget}s ran out after 1 attempt(s) "
        "without a valid response. Last error: "
    )
    assert "Missing keys: {'key'}" in str(err.value)
    assert block.execution_stats.provider_cost == 0.25


@pytest.mark.asyncio
async def test_an_override_cannot_lengthen_the_request(monkeypatch, spend_alerts):
    """The provider layer stops at the request budget whatever ``llm_call`` is
    asked for, so the wait and the reported timeout stop there too."""
    monkeypatch.setattr(llm, "LLM_REQUEST_TIMEOUT_SECONDS", _BUDGET)
    clock = {"t": float(int(time.monotonic()))}
    monkeypatch.setattr(llm, "_now", lambda: clock["t"])

    async def cut_off_at_the_budget(**kwargs) -> llm.LLMResponse:
        clock["t"] += _BUDGET
        raise TimeoutError()

    monkeypatch.setattr(llm, "_llm_call", cut_off_at_the_budget)

    with pytest.raises(TimeoutError, match=f"exceeded {_BUDGET}s"):
        await llm.llm_call(
            credentials=llm.TEST_CREDENTIALS,
            llm_model=llm.DEFAULT_LLM_MODEL,
            prompt=[{"role": "user", "content": "Hello"}],
            max_tokens=None,
            on_abandoned=None,
            timeout_seconds=_BUDGET * 3,
        )

    assert len(spend_alerts) == 1
    assert vars(spend_alerts[0])["json_fields"]["configured_timeout_seconds"] == (
        _BUDGET
    )


@pytest.mark.asyncio
async def test_the_slice_is_what_cuts_a_silent_request_off(monkeypatch):
    """The fake clock above can't see the real ``wait_for``, so this request
    never answers: the slice must be what stops it, not the whole budget."""
    monkeypatch.setattr(llm, "LLM_REQUEST_TIMEOUT_SECONDS", _BUDGET)

    async def never_answers(**kwargs) -> llm.LLMResponse:
        await asyncio.Event().wait()
        raise AssertionError("the request was never cut off")

    monkeypatch.setattr(llm, "_llm_call", never_answers)

    # A 0.1 s slice inside a 5 s bound: a wait at the whole budget would hit
    # the outer bound, whose TimeoutError carries no "exceeded" message.
    with pytest.raises(TimeoutError, match="exceeded 0.1s"):
        await asyncio.wait_for(
            llm.llm_call(
                credentials=llm.TEST_CREDENTIALS,
                llm_model=llm.DEFAULT_LLM_MODEL,
                prompt=[{"role": "user", "content": "Hello"}],
                max_tokens=None,
                on_abandoned=None,
                timeout_seconds=0.1,
            ),
            timeout=5,
        )
