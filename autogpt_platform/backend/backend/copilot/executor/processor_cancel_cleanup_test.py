"""A Stop's second cancel that lands in the turn's cleanup (#17286).

A Stop cancels the turn's task. When the turn hasn't ended within
``_CANCEL_GRACE_SECONDS``, the worker cancels it again and fails the session
itself (``CoPilotProcessor._execute``). By then the turn can already be
cleaning up, and the second cancel used to cut that cleanup short where it
landed: the background subtasks the turn launched kept running, its exit
stack was left for the garbage collector to close in another context, and a
browser turn's Chromium tree outlived it.
"""

import asyncio
import concurrent.futures
import gc
import sys
import threading
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from backend.copilot import background_execution, stream_registry, trial_reads
from backend.copilot.executor import processor
from backend.copilot.executor.utils import CoPilotExecutionEntry
from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamStatus
from backend.copilot.trial_cost_context import TrialCostContext, get_trial_cost_context
from backend.copilot.trial_reads import SharedTrialReads

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
    "reply_sees_the_stop",
    [
        # Production's: the worker's first cancel ends the reply, and its
        # second lands in the cleanup.
        pytest.param(False, id="first-cancel-ends-the-reply"),
        # The reply ends at the Stop on its own, so both of the worker's
        # cancels land in the cleanup, with no cancel already on its way out.
        pytest.param(True, id="both-cancels-reach-the-cleanup"),
    ],
)
def test_a_cancel_in_the_cleanup_waits_for_every_step(
    exec_loop, mocker, reply_sees_the_stop: bool
):
    """The worker's grace runs out while the turn's cleanup waits on
    ``finish_background_execution``'s first await."""
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
    cancel = threading.Event()
    replying = threading.Event()
    cleanup_held = threading.Event()
    release_cleanup = asyncio.Event()
    subtask = MagicMock(name="background subtask")
    turn: dict[str, asyncio.Task] = {}
    contexts: dict[str, dict] = {}

    def trial_context() -> dict:
        return {
            "shared trial reads": trial_reads._shared_reads.get(),
            "trial cost context": get_trial_cost_context(user_id),
        }

    async def reply(**_kwargs):
        task = asyncio.current_task()
        assert task is not None
        turn["task"] = task
        contexts["while replying"] = trial_context()
        replying.set()
        if reply_sees_the_stop:
            cancel.set()  # the user's Stop, which the turn sees at this chunk
            yield StreamStatus(message="Working")
        await asyncio.Event().wait()  # until the Stop's cancel ends the reply
        yield StreamStatus(message="never reached")

    async def cancel_background_tasks(*_args):
        cleanup_held.set()
        await release_cleanup.wait()
        return [subtask]

    real_turn_transaction = processor.turn_transaction

    @contextmanager
    def watched_turn_transaction(entry: CoPilotExecutionEntry):
        with real_turn_transaction(entry) as trace:
            try:
                yield trace
            finally:
                # Still on the turn's own task, after its cleanup.
                contexts["after the cleanup"] = trial_context()

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
    mocker.patch.object(processor, "refusal_for", AsyncMock(return_value=None))
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        # Returns once the worker has given up on the turn and failed the
        # session itself, as production's did.
        pool.submit(proc.execute, entry, cancel, cluster_lock).result(
            timeout=_HANG_SECONDS
        )
    exec_loop.call_soon_threadsafe(release_cleanup.set)
    turn_task = turn["task"]
    asyncio.run_coroutine_threadsafe(_ended(turn_task), exec_loop).result(
        timeout=_HANG_SECONDS
    )
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

    while_replying = contexts["while replying"]
    assert isinstance(while_replying["shared trial reads"], SharedTrialReads)
    assert while_replying["trial cost context"] == cost
    assert {
        "subtasks stopped": stopped.await_args_list,
        "session completions": completed.await_args_list,
        "browser closes": browser.await_args_list,
        "turn's trial context after its cleanup": contexts.get("after the cleanup"),
        "turn cancelled": turn_cancelled,
        "errors": errors,
    } == {
        "subtasks stopped": [call([subtask])],
        "session completions": [
            # The worker's fail-close, while the cleanup was held.
            call(
                entry.session_id,
                error_message=processor.SHUTDOWN_ERROR_MESSAGE,
                turn_id=entry.turn_id,
            ),
            # The turn's own, once released.
            call(
                entry.session_id,
                error_message=stream_registry.CANCELLED_MESSAGE,
                error_code=None,
                skip_error_publish=True,
                turn_id=entry.turn_id,
            ),
        ],
        "browser closes": [call(entry.session_id)],
        "turn's trial context after its cleanup": {
            "shared trial reads": None,
            "trial cost context": None,
        },
        "turn cancelled": True,
        "errors": [],
    }


async def _ended(task: asyncio.Task) -> None:
    async with asyncio.timeout(_HANG_SECONDS):
        await asyncio.wait({task})
