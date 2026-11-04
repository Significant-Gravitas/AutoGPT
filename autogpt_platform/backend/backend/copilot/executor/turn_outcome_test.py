"""How the executor judges a turn's end and counts it, along the path a turn's
stream really takes: the engine, the heartbeat, the registry's publishing and
the processor's loop, up to ``TurnTrace.set_outcome`` and
``copilot_executor_turns_total``. Only Redis, the chat store and the engine
behind the stream are stood in for."""

import threading
from collections.abc import AsyncGenerator, Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY, generate_latest

from backend.copilot import ai_spans
from backend.copilot.background_task import BACKGROUND_TIMEOUT_MESSAGE
from backend.copilot.baseline.service import _build_natural_finish_empty_fallback_events
from backend.copilot.executor.processor import CoPilotProcessor
from backend.copilot.model import ChatSession
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamStart,
    StreamTextDelta,
    StreamToolInputAvailable,
    StreamToolOutputAvailable,
)
from backend.copilot.turn_metrics import turns_counter

from .processor_test import _make_entry, _make_log

_PROCESSOR = "backend.copilot.executor.processor"
_LABELS = ("outcome", "engine", "origin", "load_test")

Counted = Callable[[], dict[tuple[str, ...], float]]


@pytest.fixture(autouse=True)
def allowance_admits_every_turn(mocker):
    """The pickup budget gate has its own tests (processor_budget_gate_test)."""
    mocker.patch(f"{_PROCESSOR}.refusal_for", AsyncMock(return_value=None))
    mocker.patch(
        "backend.copilot.background_execution.background_task_db",
        return_value=MagicMock(cancel_background_tasks=AsyncMock(return_value=[])),
    )


@pytest.fixture(autouse=True)
def no_trial_attribution(mocker):
    store = MagicMock()
    store.get_usage_activation_state = AsyncMock(
        side_effect=lambda user_id: SimpleNamespace(
            user_id=user_id, generation=None, trial_id=None, ready=True
        )
    )
    mocker.patch(
        "backend.copilot.usage_activation.pro_activation_db", return_value=store
    )


@pytest.fixture(autouse=True)
def posthog(mocker) -> MagicMock:
    return mocker.patch.object(ai_spans.posthog_client, "capture")


def _turns_counted() -> dict[tuple[str, ...], float]:
    """Every series of ``copilot_executor_turns_total``, by its labels."""
    return {
        tuple(sample.labels[name] for name in _LABELS): sample.value
        for metric in turns_counter.collect()
        for sample in metric.samples
        if sample.name == "copilot_executor_turns_total"
    }


@pytest.fixture
def counted() -> Counted:
    """What ``copilot_executor_turns_total`` has gained since the test began,
    by labels."""
    before = _turns_counted()

    def gained() -> dict[tuple[str, ...], float]:
        return {
            labels: value - before.get(labels, 0.0)
            for labels, value in _turns_counted().items()
            if value != before.get(labels, 0.0)
        }

    return gained


async def _run(
    events: list[StreamBaseResponse],
    *,
    engine: str = "dummy",
    load_test: str | None = None,
    cancel: threading.Event | None = None,
    then_raise: BaseException | None = None,
) -> None:
    """Run one turn on *engine*, or on a load-test account's engine when
    *load_test* names one, whose stream is *events*; after them the engine
    returns, or raises *then_raise* when given."""

    async def stream(**_kwargs) -> AsyncGenerator[StreamBaseResponse, None]:
        for event in events:
            yield event
        if then_raise is not None:
            raise then_raise

    config = MagicMock(
        test_mode=engine == "dummy" and load_test is None,
        thinking_available=False,
        load_test_email_domain="previews.agpt.co",
        load_test_model_url="http://scripted-model:4010/api/v1",
    )
    with (
        patch(f"{_PROCESSOR}.ChatConfig", return_value=config),
        patch(f"{_PROCESSOR}.load_test_engine", AsyncMock(return_value=load_test)),
        patch(f"{_PROCESSOR}.resolve_use_sdk", AsyncMock(return_value=engine == "sdk")),
        patch(f"{_PROCESSOR}.stream_chat_completion_dummy", side_effect=stream),
        patch(f"{_PROCESSOR}.stream_chat_completion_baseline", side_effect=stream),
        patch(
            f"{_PROCESSOR}.sdk_service.stream_chat_completion_sdk", side_effect=stream
        ),
        patch("backend.copilot.stream_registry.publish_chunk", new=AsyncMock()),
        patch(
            "backend.copilot.stream_registry.mark_session_completed", new=AsyncMock()
        ),
        patch(
            "backend.copilot.model.get_chat_session",
            new=AsyncMock(return_value=ChatSession.new("user-1", dry_run=False)),
        ),
    ):
        await CoPilotProcessor()._execute_async(
            _make_entry(), cancel or threading.Event(), MagicMock(), _make_log()
        )


def _error() -> StreamError:
    return StreamError(
        errorText="The response stopped. Please try again.", code="idle_timeout"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ending",
    [[_error()], [_error(), StreamFinish()]],
    ids=["an error frame", "an error frame, then the stream's finish"],
)
async def test_a_turn_whose_stream_ends_on_an_error_frame_failed_with_its_code(
    ending: list[StreamBaseResponse],
    sentry_transactions: list[dict],
    posthog: MagicMock,
    counted: Counted,
) -> None:
    await _run([StreamTextDelta(id="t", delta="Working on it"), *ending])

    [turn] = sentry_transactions
    assert turn["tags"]["copilot.turn.outcome"] == "failed"
    assert turn["contexts"]["trace"]["status"] == "internal_error"
    properties = posthog.call_args.args[2]
    assert properties["$ai_is_error"] is True
    assert properties["$ai_error"] == "idle_timeout"
    assert counted() == {("failed", "dummy", "chat", ""): 1.0}


@pytest.mark.asyncio
async def test_a_baseline_turn_ending_on_its_empty_completion_notice_failed(
    sentry_transactions: list[dict], posthog: MagicMock, counted: Counted
) -> None:
    """The baseline engine streams this notice where the SDK engine streams an
    ``empty_completion`` error frame: either way the user got no reply."""
    notice, _ = _build_natural_finish_empty_fallback_events("")

    await _run([StreamStart(messageId="m-1"), *notice], engine="baseline")

    [turn] = sentry_transactions
    assert turn["tags"]["copilot.turn.outcome"] == "failed"
    assert turn["contexts"]["trace"]["status"] == "internal_error"
    assert "copilot.turn.ttft_ms" not in turn["contexts"]["trace"]["data"]
    properties = posthog.call_args.args[2]
    assert properties["$ai_is_error"] is True
    assert properties["$ai_error"] == "empty_completion"
    assert counted() == {("failed", "baseline", "chat", ""): 1.0}


@pytest.mark.asyncio
async def test_a_turn_whose_model_went_on_from_a_tool_error_completed(
    sentry_transactions: list[dict], posthog: MagicMock, counted: Counted
) -> None:
    await _run(
        [
            StreamToolInputAvailable(toolCallId="c-1", toolName="run_block"),
            StreamToolOutputAvailable(
                toolCallId="c-1",
                toolName="run_block",
                output={"error": "The block's input was invalid."},
                success=False,
            ),
            StreamTextDelta(id="t", delta="That input was invalid, so I fixed it."),
        ],
        engine="sdk",
    )

    [turn] = sentry_transactions
    assert turn["tags"]["copilot.turn.outcome"] == "completed"
    assert posthog.call_args.args[2]["$ai_is_error"] is False
    assert counted() == {("completed", "sdk", "chat", ""): 1.0}


@pytest.mark.asyncio
async def test_a_turn_its_user_stopped_is_cancelled(
    sentry_transactions: list[dict], counted: Counted
) -> None:
    cancel = threading.Event()
    cancel.set()

    await _run([StreamTextDelta(id="t", delta="Working on it")], cancel=cancel)

    [turn] = sentry_transactions
    assert turn["tags"]["copilot.turn.outcome"] == "cancelled"
    assert counted() == {("cancelled", "dummy", "chat", ""): 1.0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("load_test", "engine"),
    [("dummy", "dummy"), ("scripted", "sdk")],
    ids=["on the dummy engine", "on the scripted model"],
)
async def test_a_load_test_accounts_turn_is_counted_as_one(
    load_test: str, engine: str, counted: Counted
) -> None:
    await _run([StreamTextDelta(id="t", delta="Done.")], load_test=load_test)

    assert counted() == {("completed", engine, "chat", load_test): 1.0}


@pytest.mark.asyncio
async def test_a_turn_that_raises_after_its_error_is_counted_once(
    sentry_transactions: list[dict], counted: Counted
) -> None:
    """Its outcome is set before the raise leaves the turn, and the raise
    neither changes it nor counts the turn a second time."""
    with pytest.raises(RuntimeError, match="the engine broke after its error"):
        await _run(
            [_error()], then_raise=RuntimeError("the engine broke after its error")
        )

    [turn] = sentry_transactions
    assert turn["tags"]["copilot.turn.outcome"] == "failed"
    assert counted() == {("failed", "dummy", "chat", ""): 1.0}


@pytest.mark.asyncio
async def test_a_turn_refused_before_its_engine_is_counted_under_no_engine(
    sentry_transactions: list[dict], posthog: MagicMock, counted: Counted
) -> None:
    """A banned owner's turn never reaches a model: PostHog's LLM analytics
    gets nothing, and the counter keeps it apart from model turns."""
    with patch(f"{_PROCESSOR}.is_user_banned", new=AsyncMock(return_value=True)):
        await _run([])

    [turn] = sentry_transactions
    assert turn["tags"]["copilot.turn.outcome"] == "failed"
    posthog.assert_not_called()
    assert counted() == {("failed", "none", "chat", ""): 1.0}


@pytest.mark.asyncio
async def test_a_background_pickup_another_worker_claimed_is_not_counted(
    posthog: MagicMock, counted: Counted
) -> None:
    """A redelivered background turn that another worker already claimed ends
    here without running; the worker that runs it counts it."""
    with patch(
        f"{_PROCESSOR}.claim_background_execution",
        new=AsyncMock(return_value=False),
    ):
        await _run([StreamTextDelta(id="t", delta="Done.")])

    posthog.assert_not_called()
    assert counted() == {}


@pytest.mark.asyncio
async def test_an_unclaimed_pickup_whose_cleanup_raises_is_not_counted(
    posthog: MagicMock, counted: Counted
) -> None:
    """Closing a pickup another worker owns can fail. The turn is still that
    worker's to count, so this one counts nothing."""

    async def claimed_elsewhere_then_cleanup_fails(entry, *, on_claimed_elsewhere):
        on_claimed_elsewhere()
        raise ConnectionError("database unavailable")

    with (
        patch(
            f"{_PROCESSOR}.claim_background_execution",
            new=claimed_elsewhere_then_cleanup_fails,
        ),
        pytest.raises(ConnectionError),
    ):
        await _run([StreamTextDelta(id="t", delta="Done.")])

    posthog.assert_not_called()
    assert counted() == {}


@pytest.mark.asyncio
async def test_a_background_turn_refused_at_admission_is_counted_failed(
    sentry_transactions: list[dict], counted: Counted
) -> None:
    """Admitting it raised before the turn could run or set an outcome of its
    own, and its user is told it failed, so it failed."""
    with (
        patch(
            f"{_PROCESSOR}.claim_background_execution",
            new=AsyncMock(side_effect=TimeoutError(BACKGROUND_TIMEOUT_MESSAGE)),
        ),
        pytest.raises(TimeoutError),
    ):
        await _run([])

    [turn] = sentry_transactions
    assert turn["tags"]["copilot.turn.outcome"] == "failed"
    assert counted() == {("failed", "none", "chat", ""): 1.0}


@pytest.mark.asyncio
async def test_a_load_test_accounts_turn_refused_at_admission_is_counted_as_one(
    counted: Counted,
) -> None:
    """Admission can end a turn before it runs; a load-test account's turn
    is still told apart from people's, so it never counts as theirs."""
    with (
        patch(
            f"{_PROCESSOR}.claim_background_execution",
            new=AsyncMock(side_effect=TimeoutError(BACKGROUND_TIMEOUT_MESSAGE)),
        ),
        pytest.raises(TimeoutError),
    ):
        await _run([], engine="sdk", load_test="scripted")

    assert counted() == {("failed", "none", "chat", "scripted"): 1.0}


@pytest.mark.asyncio
async def test_the_turn_counter_is_served_with_the_executors_metrics() -> None:
    """``CoPilotExecutor.run`` serves the default registry, the one its gauges
    are in, so the counter is scraped with them."""
    await _run([StreamTextDelta(id="t", delta="Done.")])

    served = generate_latest(REGISTRY).decode()
    assert "copilot_executor_active_tasks" in served
    # The exposition format lists a series' labels in alphabetical order.
    assert (
        'copilot_executor_turns_total{engine="dummy",load_test="",'
        'origin="chat",outcome="completed"}'
    ) in served


def test_peoples_turn_series_are_exported_at_zero_before_any_turn() -> None:
    # increase() can't count a series' first sample, so a failed turn's
    # series must already exist when the first one fails.
    served = generate_latest(REGISTRY).decode()
    for outcome in ("completed", "cancelled", "failed"):
        assert (
            "copilot_executor_turns_total{"
            f'engine="sdk",load_test="",origin="chat",outcome="{outcome}"}}'
        ) in served
