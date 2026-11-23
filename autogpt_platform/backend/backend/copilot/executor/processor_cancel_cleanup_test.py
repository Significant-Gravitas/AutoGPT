"""A Stop's second cancel that lands in the turn's cleanup (#17286).

A Stop cancels the turn's task. When the turn hasn't ended within
``_CANCEL_GRACE_SECONDS``, the worker cancels it again and fails the session
itself (``CoPilotProcessor._execute``). By then the turn can already be
cleaning up, and the second cancel used to cut that cleanup short where it
landed: the background subtasks the turn launched kept running, its exit
stack was left for the garbage collector to close in another context (which
also skipped a credit-billed turn's last charge), and a browser turn's
Chromium tree outlived it.
"""

import asyncio
import concurrent.futures
import gc
import sys
import threading
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from backend.copilot import (
    background_execution,
    credit_billing,
    stream_registry,
    trial_reads,
)
from backend.copilot.credit_billing import (
    CreditBilledTurn,
    credit_billed_turn,
    record_credit_billed_cost,
)
from backend.copilot.executor import processor
from backend.copilot.executor.utils import CoPilotExecutionEntry
from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamStatus
from backend.copilot.tree import MICRODOLLARS_PER_CREDIT
from backend.copilot.trial_cost_context import TrialCostContext, get_trial_cost_context
from backend.copilot.trial_reads import SharedTrialReads
from backend.copilot.turn_pause import TurnPause

# The longest any one wait here may take. The test waits on events, never on
# time, so this only ends a run that hangs.
_HANG_SECONDS = 30


@pytest.fixture
def exec_loop():
    """The worker's long-lived loop, on a thread of its own as the processor
    runs it."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=_HANG_SECONDS)
        loop.close()


@pytest.mark.parametrize(
    "reply_sees_the_stop, held_at",
    [
        # Production's: the worker's first cancel ends the reply, and its
        # second lands while the cleanup stops the turn's subtasks.
        pytest.param(False, "subtasks", id="first-cancel-ends-the-reply"),
        # The reply ends at the Stop on its own, so both of the worker's
        # cancels land in the cleanup, with no cancel already on its way out.
        pytest.param(True, "subtasks", id="both-cancels-reach-the-cleanup"),
        # The second cancel lands while the turn is charged its credits.
        pytest.param(False, "charge", id="second-cancel-during-the-charge"),
    ],
)
def test_a_cancel_in_the_cleanup_waits_for_every_step(
    exec_loop, mocker, reply_sees_the_stop: bool, held_at: str
):
    """The worker's grace runs out while the cleanup of a credit-billed turn
    waits on a step: ``finish_background_execution``'s first await, or the
    debit that charges the turn."""
    user_id = "user-stop"
    cost = TrialCostContext(user_id=user_id, trial_id="trial-1")
    entry = CoPilotExecutionEntry(
        session_id="session-stop",
        turn_id="turn-stop",
        user_id=user_id,
        message="hi",
        is_user_message=True,
        request_arrival_at=0.0,
        cost_context=cost,
    )
    pause = TurnPause(reason="daily")
    billed = CreditBilledTurn(
        user_id=user_id, session_id=entry.session_id, cap_credits=10, refusal=pause
    )
    cancel = threading.Event()
    replying = threading.Event()
    cleanup_held = threading.Event()
    release_cleanup = asyncio.Event()
    subtask = MagicMock(name="background subtask")
    charged: list[int] = []
    turn: dict[str, asyncio.Task] = {}
    seen: dict = {}

    def turn_context() -> dict:
        return {
            "shared trial reads": trial_reads._shared_reads.get(),
            "trial cost context": get_trial_cost_context(user_id),
            "credit-billed turn": credit_billed_turn(user_id),
        }

    async def hold() -> None:
        cleanup_held.set()
        await release_cleanup.wait()

    async def reply(**_kwargs):
        task = asyncio.current_task()
        assert task is not None
        turn["task"] = task
        seen["context while replying"] = turn_context()
        # Half a credit: nothing is charged until the turn ends.
        await record_credit_billed_cost(user_id, MICRODOLLARS_PER_CREDIT // 2)
        replying.set()
        if reply_sees_the_stop:
            cancel.set()  # the user's Stop, which the turn sees at this chunk
            yield StreamStatus(message="Working")
        await asyncio.Event().wait()  # until the Stop's cancel ends the reply
        yield StreamStatus(message="never reached")

    async def cancel_background_tasks(*_args):
        if held_at == "subtasks":
            await hold()
        return [subtask]

    async def spend_credits(*, cost: int, **_kwargs) -> None:
        if held_at == "charge":
            await hold()
        charged.append(cost)

    real_turn_transaction = processor.turn_transaction

    @contextmanager
    def watched_turn_transaction(entry: CoPilotExecutionEntry):
        with real_turn_transaction(entry) as trace:
            try:
                yield trace
            finally:
                # Still on the turn's own task, after its cleanup.
                seen["context after the cleanup"] = turn_context()
                seen["credits charged by the turn's end"] = list(charged)

    def refresh() -> None:
        """The worker calls this once a second while it waits on the turn."""
        if not cancel.is_set():
            if replying.is_set():
                cancel.set()  # the user's Stop: the next poll cancels the turn
            return
        # That poll cancelled the turn, and the next finds the grace run out:
        # by then, the turn's cleanup must be waiting where production's was.
        cleanup_held.wait(_HANG_SECONDS)

    errors: list[str] = []
    mocker.patch.object(
        sys,
        "unraisablehook",
        lambda raised: errors.append(f"{raised.exc_type.__name__}: {raised.exc_value}"),
    )
    exec_loop.set_exception_handler(
        lambda _loop, context: errors.append(
            f"{context['message']}: {context.get('exception')!r}"
        )
    )
    mocker.patch.object(processor, "_CANCEL_GRACE_SECONDS", 0.1)
    mocker.patch.object(processor, "account_refusal", AsyncMock(return_value=None))
    # The allowance is used up, and the AutoPilot block's turn runs on credits.
    mocker.patch.object(processor, "refusal_for", AsyncMock(return_value=pause))
    mocker.patch.object(processor, "bill_to_credits", AsyncMock(return_value=billed))
    mocker.patch.object(
        credit_billing,
        "credit_db",
        return_value=MagicMock(spend_credits=spend_credits),
    )
    mocker.patch.object(processor, "ChatConfig", return_value=MagicMock(test_mode=True))
    mocker.patch.object(processor, "stream_chat_completion_dummy", return_value=None)
    mocker.patch.object(processor, "turn_transaction", watched_turn_transaction)
    mocker.patch(
        "backend.copilot.model.get_chat_session",
        AsyncMock(return_value=ChatSession.new(user_id, dry_run=False)),
    )
    mocker.patch.object(stream_registry, "publish_chunk", AsyncMock())
    mocker.patch.object(stream_registry, "stream_and_publish", reply)
    mocker.patch.object(
        background_execution,
        "background_task_db",
        return_value=MagicMock(cancel_background_tasks=cancel_background_tasks),
    )
    stopped = mocker.patch.object(
        background_execution, "stop_cancelled_subtasks", AsyncMock()
    )
    completed = mocker.patch.object(
        stream_registry, "mark_session_completed", AsyncMock(return_value=True)
    )
    browser = mocker.patch.object(
        processor, "close_browser_daemon", AsyncMock(return_value=True)
    )
    cluster_lock = MagicMock()
    cluster_lock.refresh.side_effect = refresh
    proc = processor.CoPilotProcessor()
    proc.execution_loop = exec_loop

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            worker = pool.submit(proc.execute, entry, cancel, cluster_lock)
            try:
                # Returns once the worker has given up on the turn and failed
                # the session itself, as production's did.
                worker.result(timeout=_HANG_SECONDS)
            finally:
                # Then the held step goes on. When the wait above fails, the
                # Stop also has the worker give up before the pool shuts down.
                cancel.set()
                exec_loop.call_soon_threadsafe(release_cleanup.set)
    finally:
        # Nothing of the turn outlives the test, even when it fails above.
        if "task" in turn:
            asyncio.run_coroutine_threadsafe(_ended(turn["task"]), exec_loop).result(
                timeout=_HANG_SECONDS
            )
    turn_task = turn["task"]
    turn_cancelled = turn_task.cancelled()
    # In production the garbage collector closed the abandoned exit stack,
    # in another context, twelve minutes later.
    del turn_task
    turn.clear()
    gc.collect()
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0), exec_loop).result(
        timeout=_HANG_SECONDS
    )
    gc.collect()

    while_replying = seen["context while replying"]
    assert isinstance(while_replying["shared trial reads"], SharedTrialReads)
    assert while_replying["trial cost context"] == cost
    assert while_replying["credit-billed turn"] is billed
    fail_close = call(
        entry.session_id,
        error_message=processor.SHUTDOWN_ERROR_MESSAGE,
        turn_id=entry.turn_id,
    )
    turns_own = call(
        entry.session_id,
        error_message=stream_registry.CANCELLED_MESSAGE,
        error_code=None,
        skip_error_publish=True,
        turn_id=entry.turn_id,
    )
    assert {
        "subtasks stopped": stopped.await_args_list,
        "session completions": completed.await_args_list,
        "credits charged by the turn's end": seen.get(
            "credits charged by the turn's end"
        ),
        "credits charged in all": charged,
        "browser closes": browser.await_args_list,
        "turn's context after its cleanup": seen.get("context after the cleanup"),
        "turn cancelled": turn_cancelled,
        "errors": errors,
    } == {
        "subtasks stopped": [call([subtask])],
        # The worker fails the session while the cleanup is held, which is
        # before the turn's own completion when the subtasks are held, and
        # after it when the charge, which comes later, is.
        "session completions": (
            [fail_close, turns_own]
            if held_at == "subtasks"
            else [turns_own, fail_close]
        ),
        # Half a credit spent, rounded up when the turn ends.
        "credits charged by the turn's end": [1],
        "credits charged in all": [1],
        "browser closes": [call(entry.session_id)],
        "turn's context after its cleanup": {
            "shared trial reads": None,
            "trial cost context": None,
            "credit-billed turn": None,
        },
        "turn cancelled": True,
        "errors": [],
    }


async def _ended(task: asyncio.Task) -> None:
    async with asyncio.timeout(_HANG_SECONDS):
        await asyncio.wait({task})
