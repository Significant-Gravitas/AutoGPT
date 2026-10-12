"""A turn's Sentry trace is handed over in its queue message, from the
transaction that caused the turn to the turn's own transaction."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.executor import utils
from backend.copilot.executor.utils import CoPilotExecutionEntry
from backend.copilot.tree import root_envelope
from backend.util import tracing


def _published_entry(publish: AsyncMock) -> CoPilotExecutionEntry:
    return CoPilotExecutionEntry.model_validate_json(
        publish.await_args.kwargs["message"]
    )


@pytest.fixture
def copilot_queue():
    queue = MagicMock(publish_message=AsyncMock())
    with patch(
        "backend.util.clients.get_async_copilot_queue",
        new=AsyncMock(return_value=queue),
    ):
        yield queue


async def _enqueue(**kwargs) -> None:
    await utils.enqueue_copilot_turn(
        session_id="s1",
        user_id="u1",
        message="hi",
        turn_id="t1",
        envelope=root_envelope("t1"),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_a_turn_enqueued_inside_a_transaction_continues_its_trace(
    sentry_transactions, copilot_queue
) -> None:
    before = time.time()
    with tracing.start_transaction("http.server", "/stream") as request:
        await _enqueue()

    entry = _published_entry(copilot_queue.publish_message)
    assert entry.trace_headers["sentry-trace"].startswith(request.trace_id)
    assert "baggage" in entry.trace_headers
    assert before <= entry.enqueued_at <= time.time()


@pytest.mark.asyncio
async def test_an_untraced_enqueue_hands_over_no_trace(
    sentry_transactions, copilot_queue
) -> None:
    await _enqueue()

    assert _published_entry(copilot_queue.publish_message).trace_headers == {}


def test_an_entry_from_before_the_trace_fields_still_decodes() -> None:
    legacy = '{"session_id": "s1", "user_id": "u1", "message": "hi"}'

    entry = CoPilotExecutionEntry.model_validate_json(legacy)

    assert entry.trace_headers == {}
    assert entry.enqueued_at == 0.0


@pytest.mark.parametrize(
    "dispatch_kwargs, continues_caller",
    [
        ({}, True),
        ({"root": True}, False),
        ({"envelope": root_envelope("held")}, False),
    ],
    ids=["derived-from-caller", "minted-root", "re-admitted"],
)
@pytest.mark.asyncio
async def test_dispatch_continues_only_the_trace_of_the_turns_cause(
    sentry_transactions, dispatch_kwargs: dict, continues_caller: bool
) -> None:
    """A promoted or woken turn is dispatched from inside whatever turn just
    ended, which did not cause it, so only a turn whose envelope is derived
    from the caller joins the caller's trace."""
    enqueue = AsyncMock()
    with (
        patch.object(
            utils,
            "_admitted_turn_envelope",
            new=AsyncMock(return_value=root_envelope("t1")),
        ),
        patch.object(utils, "readmit_turn", new=AsyncMock()),
        patch("backend.copilot.stream_registry.create_session", new=AsyncMock()),
        patch.object(utils, "enqueue_copilot_turn", new=enqueue),
    ):
        with tracing.start_transaction("copilot.turn", "copilot.turn"):
            await utils.dispatch_turn(
                MagicMock(),
                session_id="s1",
                user_id="u1",
                turn_id="t1",
                message="hi",
                **dispatch_kwargs,
            )

    trace_headers = enqueue.await_args.kwargs["trace_headers"]
    assert trace_headers == (None if continues_caller else {})
