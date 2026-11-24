"""A Stop's second cancel must not cut a turn's cleanup short (#17286).

A Stop cancels the turn's task. When the turn hasn't ended within
``_CANCEL_GRACE_SECONDS``, the worker cancels it again and fails the session
itself (``CoPilotProcessor._execute``), and that second cancel can land in
the turn's cleanup. Every cleanup step must still run to its end: the
subtasks the turn launched are stopped, its completion and scheduled run are
recorded, a credit-billed turn is charged, its context variables are reset on
its own task, and its browser is closed, unless a newer turn of the chat runs
on this pod by then (#17293). The turn still ends cancelled.
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
    after_turn,
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

# The longest any one wait here may take. The tests wait on events, never on
# time, so this only ends a run that hangs.
_HANG_SECONDS = 30
_USER_ID = "user-stop"
_SESSION_ID = "session-stop"
_TURN_ID = "turn-stop"
_NEWER_TURN_ID = "turn-newer"
_SUBTASK = "the turn's background subtask"


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
        pytest.param(False, "charge", id="second-cancel-during-the-charge"),
        pytest.param(False, "record", id="second-cancel-during-the-record"),
        # On a route that leases its credential, so not billed to credits.
        pytest.param(
            False, "lease release", id="second-cancel-during-the-lease-release"
        ),
    ],
)
def test_a_cancel_in_the_cleanup_waits_for_every_step(
    exec_loop, mocker, reply_sees_the_stop: bool, held_at: str
):
    """The worker's grace runs out while the cleanup of a scheduled browser
    turn, billed to credits or on a leased credential, waits on a step."""
    route = "microsoft 365" if held_at == "lease release" else "credits"
    seen = _stop_a_turn(
        exec_loop,
        mocker,
        reply_sees_the_stop=reply_sees_the_stop,
        held_at=held_at,
        route=route,
    )

    assert seen == _expected(
        reply_sees_the_stop=reply_sees_the_stop, held_at=held_at, route=route
    )


def test_an_error_from_a_later_step_does_not_drop_the_held_cancel(exec_loop, mocker):
    """The browser close fails after the second cancel was held: the cancel
    wins, and the turn still ends cancelled."""
    seen = _stop_a_turn(exec_loop, mocker, browser_close_fails=True)

    assert seen == _expected()


@pytest.mark.parametrize(
    "newer_turn_runs_here, browser_closes",
    [
        # The browser is the newer turn's now, mid-step perhaps.
        pytest.param(True, [], id="newer-turn-on-this-pod"),
        # This pod's browser is still the stopped turn's: left open, nothing
        # would ever close it.
        pytest.param(False, [call(_SESSION_ID)], id="newer-turn-on-another-pod"),
    ],
)
def test_a_stopped_turn_leaves_the_browser_of_a_newer_turn_here(
    exec_loop, mocker, newer_turn_runs_here: bool, browser_closes: list
):
    """The chat's next turn starts while the stopped turn still cleans up."""
    seen = _stop_a_turn(
        exec_loop,
        mocker,
        chat_turn_at_close=_NEWER_TURN_ID,
        newer_turn_runs_here=newer_turn_runs_here,
    )

    assert seen == _expected(browser_closes=browser_closes)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "daemon_here, chat_turn, newer_turn_runs_here, leaves_it",
    [
        # Nothing of the chat's on this pod: its turn isn't even read.
        pytest.param(False, _NEWER_TURN_ID, True, False, id="no-daemon-here"),
        pytest.param(True, _TURN_ID, False, False, id="still-this-turns"),
        pytest.param(True, None, False, False, id="no-turn-recorded"),
        pytest.param(True, _NEWER_TURN_ID, True, True, id="newer-turn-here"),
        pytest.param(True, _NEWER_TURN_ID, False, False, id="newer-turn-elsewhere"),
        pytest.param(True, RuntimeError("redis down"), True, False, id="unreadable"),
    ],
)
async def test_only_a_newer_turn_here_keeps_the_stopped_turns_browser(
    mocker,
    daemon_here: bool,
    chat_turn: "str | None | Exception",
    newer_turn_runs_here: bool,
    leaves_it: bool,
):
    entry = CoPilotExecutionEntry(
        session_id=_SESSION_ID, turn_id=_TURN_ID, user_id=_USER_ID, message="hi"
    )
    mocker.patch.object(processor, "browser_daemon_may_run", return_value=daemon_here)
    get_session = mocker.patch.object(
        stream_registry,
        "get_session",
        AsyncMock(
            side_effect=chat_turn if isinstance(chat_turn, Exception) else None,
            return_value=(
                None
                if chat_turn is None or isinstance(chat_turn, Exception)
                else _active_session(chat_turn)
            ),
        ),
    )
    mocker.patch.object(
        processor.after_turn, "running_here", return_value=newer_turn_runs_here
    )

    assert await processor._newer_turn_runs_here(entry) is leaves_it
    assert get_session.await_count == (1 if daemon_here else 0)


def _expected(
    *,
    reply_sees_the_stop: bool = False,
    held_at: str = "subtasks",
    route: str = "credits",
    browser_closes: list | None = None,
) -> dict:
    """What a stopped turn's cleanup must have done (see _stop_a_turn)."""
    fail_close = call(
        _SESSION_ID, error_message=processor.SHUTDOWN_ERROR_MESSAGE, turn_id=_TURN_ID
    )
    turns_own = call(
        _SESSION_ID,
        error_message=stream_registry.CANCELLED_MESSAGE,
        error_code=None,
        skip_error_publish=True,
        turn_id=_TURN_ID,
    )
    return {
        "cancels in flight when the cleanup was held": 0 if reply_sees_the_stop else 1,
        "subtasks stopped": [call([_SUBTASK])],
        # The worker fails the session while the cleanup is held: before the
        # turn's own completion when a step before it is held, after it when
        # a later step is.
        "session completions": (
            [fail_close, turns_own]
            if held_at in ("lease release", "subtasks")
            else [turns_own, fail_close]
        ),
        # Half a credit spent, rounded up when the turn ends.
        "credits charged by the turn's end": [1] if route == "credits" else [],
        "credits charged in all": [1] if route == "credits" else [],
        "lease releases": [True] if route == "microsoft 365" else [],
        "scheduled run recorded": [stream_registry.CANCELLED_MESSAGE],
        "browser closes": (
            [call(_SESSION_ID)] if browser_closes is None else browser_closes
        ),
        "turn's context after its cleanup": {
            "shared trial reads": None,
            "trial cost context": None,
            "credit-billed turn": None,
        },
        "turn cancelled": True,
        "different-Context errors": [],
    }


def _active_session(turn_id: str) -> stream_registry.ActiveSession:
    """The chat's stream as the registry has it, on turn *turn_id*."""
    return stream_registry.ActiveSession(
        session_id=_SESSION_ID,
        user_id=_USER_ID,
        tool_call_id="chat_stream",
        tool_name="chat",
        turn_id=turn_id,
    )


def _stop_a_turn(
    exec_loop: asyncio.AbstractEventLoop,
    mocker,
    *,
    reply_sees_the_stop: bool = False,
    held_at: str = "subtasks",
    route: str = "credits",
    chat_turn_at_close: str = _TURN_ID,
    newer_turn_runs_here: bool = False,
    browser_close_fails: bool = False,
) -> dict:
    """Run a scheduled browser turn through the worker and Stop it. On the
    ``credits`` route the AutoPilot block's turn is billed to credits; on
    ``microsoft 365`` it holds a lease on its credential. The worker's grace
    runs out while its cleanup is held at *held_at*: the lease's release, the
    subtasks' cancellation, the debit that charges the turn, or the record of
    its scheduled run. Returns what the turn and its cleanup did."""
    cost = TrialCostContext(user_id=_USER_ID, trial_id="trial-1")
    leased = route == "microsoft 365"
    provider = "microsoft_365_copilot" if leased else "platform"
    credential_id = "cred-microsoft" if leased else None
    entry = CoPilotExecutionEntry(
        session_id=_SESSION_ID,
        turn_id=_TURN_ID,
        user_id=_USER_ID,
        message="hi",
        is_user_message=True,
        request_arrival_at=0.0,
        cost_context=cost,
        llm_auth_provider=provider,
        llm_credential_id=credential_id,
    )
    pause = TurnPause(reason="daily")
    billed = CreditBilledTurn(
        user_id=_USER_ID, session_id=_SESSION_ID, cap_credits=10, refusal=pause
    )
    cancel = threading.Event()
    replying = threading.Event()
    cleanup_held = threading.Event()
    stop_seen = asyncio.Event()
    release_cleanup = asyncio.Event()
    charged: list[int] = []
    recorded: list[str | None] = []
    released: list[bool] = []
    turn: dict[str, asyncio.Task] = {}
    seen: dict = {}

    def turn_context() -> dict:
        return {
            "shared trial reads": trial_reads._shared_reads.get(),
            "trial cost context": get_trial_cost_context(_USER_ID),
            "credit-billed turn": credit_billed_turn(_USER_ID),
        }

    async def hold() -> None:
        seen["cancels in flight"] = turn["task"].cancelling()
        cleanup_held.set()
        await release_cleanup.wait()

    async def reply(**_kwargs):
        task = asyncio.current_task()
        assert task is not None
        turn["task"] = task
        seen["context while replying"] = turn_context()
        # Half a credit: nothing is charged until the turn ends.
        await record_credit_billed_cost(_USER_ID, MICRODOLLARS_PER_CREDIT // 2)
        replying.set()
        if reply_sees_the_stop:
            await stop_seen.wait()
            yield StreamStatus(message="Working")  # the turn sees the Stop here
        await asyncio.Event().wait()  # until the Stop's cancel ends the reply
        yield StreamStatus(message="never reached")

    async def cancel_background_tasks(*_args):
        if held_at == "subtasks":
            await hold()
        return [_SUBTASK]

    async def spend_credits(*, cost: int, **_kwargs) -> None:
        if held_at == "charge":
            await hold()
        charged.append(cost)

    async def record(turn_error: str | None) -> None:
        if held_at == "record":
            await hold()
        recorded.append(turn_error)

    async def release_lease() -> None:
        if held_at == "lease release":
            await hold()
        released.append(True)

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
                cancel.set()  # the user's Stop
                if reply_sees_the_stop:
                    # The reply sees the Stop at its next chunk and ends on its
                    # own. Held here until the cleanup is, the worker sees the
                    # Stop only after that, so both its cancels land there.
                    exec_loop.call_soon_threadsafe(stop_seen.set)
                    cleanup_held.wait(_HANG_SECONDS)
            return
        # That poll cancelled the turn, and the next finds the grace run out:
        # by then, the turn's cleanup must be waiting where production's was.
        cleanup_held.wait(_HANG_SECONDS)

    # What earlier tests left for the garbage collector is not this turn's.
    gc.collect()
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
        credit_billing, "credit_db", return_value=MagicMock(spend_credits=spend_credits)
    )
    mocker.patch.object(
        processor.ScheduledTurnWatch,
        "for_entry",
        return_value=MagicMock(record=record),
    )
    mocker.patch.object(processor, "ChatConfig", return_value=MagicMock(test_mode=True))
    mocker.patch.object(processor, "stream_chat_completion_dummy", return_value=None)
    mocker.patch.object(processor, "turn_transaction", watched_turn_transaction)
    mocker.patch(
        "backend.copilot.model.get_chat_session",
        AsyncMock(
            return_value=ChatSession.new(
                _USER_ID,
                dry_run=False,
                llm_auth_provider=provider,
                llm_credential_id=credential_id,
            )
        ),
    )
    lease = MagicMock(release=release_lease)
    mocker.patch.object(
        processor,
        "IntegrationCredentialsManager",
        return_value=MagicMock(acquire_lease=AsyncMock(return_value=lease)),
    )
    mocker.patch.object(
        processor, "is_microsoft_365_copilot_credential", return_value=True
    )
    mocker.patch.object(
        processor, "stream_chat_completion_microsoft_365", return_value=None
    )
    mocker.patch.object(stream_registry, "publish_chunk", AsyncMock())
    mocker.patch.object(stream_registry, "stream_and_publish", reply)
    # The turn used the browser, and the chat is on *chat_turn_at_close* when
    # its cleanup reaches the close.
    mocker.patch.object(processor, "browser_daemon_may_run", return_value=True)
    mocker.patch.object(
        stream_registry,
        "get_session",
        AsyncMock(return_value=_active_session(chat_turn_at_close)),
    )
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
        processor,
        "close_browser_daemon",
        AsyncMock(
            return_value=True,
            side_effect=(
                RuntimeError("browser close failed") if browser_close_fails else None
            ),
        ),
    )
    cluster_lock = MagicMock()
    cluster_lock.refresh.side_effect = refresh
    proc = processor.CoPilotProcessor()
    proc.execution_loop = exec_loop

    if newer_turn_runs_here:
        after_turn.turn_started(_NEWER_TURN_ID)
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
                exec_loop.call_soon_threadsafe(stop_seen.set)
                exec_loop.call_soon_threadsafe(release_cleanup.set)
    finally:
        # Nothing of the turn outlives the test, even when it fails above.
        if "task" in turn:
            asyncio.run_coroutine_threadsafe(_ended(turn["task"]), exec_loop).result(
                timeout=_HANG_SECONDS
            )
        if newer_turn_runs_here:
            after_turn.turn_finished(_NEWER_TURN_ID)
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
    assert while_replying["credit-billed turn"] is (None if leased else billed)
    return {
        "cancels in flight when the cleanup was held": seen.get("cancels in flight"),
        "subtasks stopped": stopped.await_args_list,
        "session completions": completed.await_args_list,
        "credits charged by the turn's end": seen.get(
            "credits charged by the turn's end"
        ),
        "credits charged in all": charged,
        "lease releases": released,
        "scheduled run recorded": recorded,
        "browser closes": browser.await_args_list,
        "turn's context after its cleanup": seen.get("context after the cleanup"),
        "turn cancelled": turn_cancelled,
        "different-Context errors": [e for e in errors if "different Context" in e],
    }


async def _ended(task: asyncio.Task) -> None:
    async with asyncio.timeout(_HANG_SECONDS):
        await asyncio.wait({task})
