"""Real Postgres/Redis handoffs, with provider calls and Rabbit publish replaced."""

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.models import ChatMessage as DbMessage
from prisma.models import User
from pydantic import BaseModel, ConfigDict

from backend.api.features.chat.routes import cancel_session_task
from backend.copilot import (
    background_delivery,
    background_execution,
    background_subtasks,
    background_task_db,
    model,
    stream_registry,
    tree,
    turn_queue,
)
from backend.copilot.active_turns import TurnSlot
from backend.copilot.background_execution import (
    claim_background_execution,
    finish_background_execution,
)
from backend.copilot.background_task import (
    BACKGROUND_TIMEOUT_MESSAGE,
    BackgroundSessionState,
    BackgroundTask,
)
from backend.copilot.context import set_current_turn_id, set_execution_context
from backend.copilot.executor import processor, utils
from backend.copilot.executor.utils import CoPilotExecutionEntry, CoPilotLogMetadata
from backend.copilot.llm_attribution import set_turn_trace_id
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    append_and_save_message,
    create_chat_session,
)
from backend.copilot.pending_message_helpers import persist_pending_as_user_rows
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.response_model import StreamError, StreamTextDelta
from backend.copilot.tool_call_context import tool_call_context
from backend.copilot.tools.helpers import _permission_denied
from backend.copilot.tree import (
    TreeRefusal,
    TurnEnvelope,
    get_tree_ledger,
    root_envelope,
)
from backend.copilot.trial_cost_context import TrialCostContext
from backend.copilot.turn_pause import TurnPause
from backend.data import db
from backend.data.db_accessors import chat_db
from backend.data.redis_client import get_redis_async

pytestmark = pytest.mark.asyncio(loop_scope="session")


class World(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_id: str
    parent: ChatSession
    envelope: TurnEnvelope
    published: AsyncMock

    async def launch(
        self,
        call_id: str = "launch",
        permissions: CopilotPermissions | None = None,
        trace_id: str | None = None,
    ) -> BackgroundTask:
        set_execution_context(
            self.user_id, self.parent, permissions=permissions, envelope=self.envelope
        )
        set_turn_trace_id(trace_id or self.envelope.tree_id)
        set_current_turn_id(self.envelope.tree_id)
        with tool_call_context(call_id):
            response = await background_subtasks.start_background_subtask(
                self.user_id, self.parent, "Find the answer"
            )
        task = await background_task_db.get_background_task(response.id, self.user_id)
        assert task
        return task

    def entry(self, session_id: str) -> CoPilotExecutionEntry:
        calls = [
            c.kwargs
            for c in self.published.await_args_list
            if c.kwargs["session_id"] == session_id
        ]
        assert len(calls) == 1
        return CoPilotExecutionEntry(
            **calls[0],
            cost_context=TrialCostContext(user_id=self.user_id, trial_id=None),
        )

    async def finish_child(self, task: BackgroundTask) -> None:
        entry = self.entry(task.child_session_id)
        assert await claim_background_execution(entry)
        await append_and_save_message(
            task.child_session_id,
            ChatMessage(role="assistant", content="The answer is 42."),
        )
        await finish_background_execution(entry, None)
        await stream_registry.mark_session_completed(
            task.child_session_id, turn_id=task.child_turn_id
        )

    async def finish_parent(self) -> None:
        await stream_registry.mark_session_completed(
            self.parent.session_id, turn_id=self.envelope.tree_id
        )


async def test_result_outlives_parent_and_wakes_once_with_original_limits(world: World):
    task = await world.launch()
    await world.finish_parent()
    await world.finish_child(task)
    await background_delivery.reconcile_background_subtasks()
    await background_delivery.reconcile_background_subtasks()
    wake = world.entry(world.parent.session_id)
    assert wake.turn_id == task.wake_id
    assert task.child_session_id in wake.message
    assert "get_sub_session_result" in wake.message
    # The chat carries on under the launching turn's own envelope.
    assert wake.envelope == task.context.envelope
    assert wake.unattended
    assert wake.envelope.spend_session_id == world.parent.session_id
    assert await claim_background_execution(wake)
    assert not await claim_background_execution(wake)
    rows = await DbMessage.prisma().find_many(
        where={"sessionId": world.parent.session_id}
    )
    assert len([r for r in rows if r.id == task.wake_id]) == 1


async def test_completion_waits_for_a_busy_parent(world: World):
    task = await world.launch()
    await world.finish_child(task)
    await background_delivery.reconcile_background_subtasks()
    assert len(world.published.await_args_list) == 1
    await world.finish_parent()
    await background_delivery.reconcile_background_subtasks()
    assert world.entry(world.parent.session_id).turn_id == task.wake_id


async def test_shared_trace_does_not_change_the_launching_turn(world: World):
    task = await world.launch(trace_id="spawning-turn-trace")
    assert task.launch_turn_id == world.envelope.tree_id
    await background_delivery.cancel_background_subtasks(
        world.user_id, world.parent.session_id, world.envelope.tree_id
    )
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.wake_suppressed and stored.outcome == "cancelled"


@pytest.mark.parametrize("blocks_exclude", [False, True])
async def test_background_child_keeps_block_restrictions_and_may_delegate(
    world: World, blocks_exclude: bool
):
    task = await world.launch(
        permissions=CopilotPermissions(
            tools=["run_sub_session", "run_capability"],
            tools_exclude=False,
            blocks=["ForbiddenBlock" if blocks_exclude else "AllowedBlock"],
            blocks_exclude=blocks_exclude,
        )
    )
    entry = CoPilotExecutionEntry.model_validate_json(
        world.entry(task.child_session_id).model_dump_json()
    )
    child = await model.get_chat_session(task.child_session_id, world.user_id)
    assert child and entry.envelope and entry.permissions
    set_execution_context(
        world.user_id, child, permissions=entry.permissions, envelope=entry.envelope
    )
    assert _permission_denied("allowed", "AllowedBlock", child.session_id) is None
    assert (
        _permission_denied("forbidden", "ForbiddenBlock", child.session_id) is not None
    )
    assert entry.envelope.permits("run_capability")
    # Like a foreground subtask, it may hand work on within the tree's depth.
    assert entry.envelope.permits("run_sub_session")


async def test_spent_subtask_tree_does_not_stop_the_chat_from_carrying_on(
    world: World,
):
    """The continuation is a turn of the chat, governed by the user's
    allowance like any other, not by the subtask's node."""
    task = await world.launch()
    await world.finish_parent()
    await world.finish_child(task)
    await (await get_tree_ledger()).charge(task.context.envelope.tree_id, 1_000_000)
    await background_delivery.reconcile_background_subtasks()
    await background_delivery.reconcile_background_subtasks()
    wake = world.entry(world.parent.session_id)
    assert wake.turn_id == task.wake_id
    assert await claim_background_execution(wake)


async def test_worker_rechecks_budget_after_queue_publish(world: World):
    task = await world.launch()
    entry = world.entry(task.child_session_id)
    await (await get_tree_ledger()).charge(task.context.envelope.tree_id, 1_000_000)
    with pytest.raises(TreeRefusal):
        await claim_background_execution(entry)
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.outcome == "failed"


async def test_usage_refusal_is_a_failed_subtask(
    world: World, monkeypatch: pytest.MonkeyPatch
):
    task = await world.launch()
    engine = MagicMock()
    monkeypatch.setattr(processor, "ChatConfig", lambda: MagicMock(test_mode=True))
    monkeypatch.setattr(processor, "stream_chat_completion_dummy", engine)
    monkeypatch.setattr(
        processor,
        "refusal_for",
        AsyncMock(
            return_value=TurnPause(
                reason="daily",
                resets_at=datetime.now(UTC) + timedelta(hours=1),
                checkpoint_saved=False,
                started=False,
            )
        ),
    )
    monkeypatch.setattr(
        processor, "close_browser_daemon", AsyncMock(return_value=False)
    )
    await processor.CoPilotProcessor()._execute_async(
        world.entry(task.child_session_id),
        threading.Event(),
        MagicMock(),
        CoPilotLogMetadata(logging.getLogger("background-test")),
    )
    engine.assert_not_called()
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.outcome == "failed"


@pytest.mark.parametrize(
    "error,outcome", [(None, "completed"), ("Operation cancelled", "cancelled")]
)
async def test_delayed_completion_preserves_the_workers_outcome(
    world: World, error: str | None, outcome: str
):
    task = await world.launch()
    await append_and_save_message(
        task.child_session_id,
        ChatMessage(role="assistant", content="The answer is 42."),
    )
    await db.execute_raw_with_schema(
        'UPDATE {schema_prefix}"ChatBackgroundTask" SET deadline=$2::timestamptz WHERE id=$1',
        task.id,
        datetime.now(UTC) - timedelta(seconds=1),
    )
    await background_task_db.finish_background_turn(
        world.user_id, task.child_session_id, task.child_turn_id, error
    )
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.outcome == outcome


@pytest.mark.parametrize("expired_before_pickup", [False, True])
async def test_worker_timeout_does_not_depend_on_the_database_clock(
    world: World, monkeypatch: pytest.MonkeyPatch, expired_before_pickup: bool
):
    task = await world.launch()
    entry = world.entry(task.child_session_id)
    assert entry.envelope
    if expired_before_pickup:
        entry.envelope = entry.envelope.model_copy(
            update={"deadline_at": datetime.now(UTC) - timedelta(seconds=1)}
        )

    async def silent_model(**_kwargs: Any):
        await asyncio.sleep(10)
        yield StreamTextDelta(id="late", delta="Too late")

    monkeypatch.setattr(processor, "background_timeout", lambda _: 0.1)
    monkeypatch.setattr(processor, "ChatConfig", lambda: MagicMock(test_mode=True))
    monkeypatch.setattr(processor, "stream_chat_completion_dummy", silent_model)
    monkeypatch.setattr(processor, "refusal_for", AsyncMock(return_value=None))
    monkeypatch.setattr(
        processor, "close_browser_daemon", AsyncMock(return_value=False)
    )
    worker = processor.CoPilotProcessor()
    worker.execution_loop = asyncio.get_running_loop()
    with (
        pytest.raises(TimeoutError, match=BACKGROUND_TIMEOUT_MESSAGE)
        if expired_before_pickup
        else nullcontext()
    ):
        await asyncio.to_thread(worker.execute, entry, threading.Event(), MagicMock())
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.outcome == "timed_out"
    active = await stream_registry.get_session(task.child_session_id)
    assert active and active.status == "failed"
    assert await chat_db().get_chat_session_status(task.child_session_id) == "idle"
    events: asyncio.Queue[stream_registry.StreamEntry] = asyncio.Queue()
    await stream_registry._replay(
        await get_redis_async(), task.child_turn_id, "0-0", events
    )
    chunks = [events.get_nowait()[1] for _ in range(events.qsize())]
    assert [c.errorText for c in chunks if isinstance(c, StreamError)] == [
        BACKGROUND_TIMEOUT_MESSAGE
    ]


async def test_child_stop_reports_cancellation_and_parent_stop_suppresses_wake(
    world: World,
):
    task = await world.launch()
    await background_delivery.cancel_background_subtasks(
        world.user_id, task.child_session_id
    )
    await world.finish_parent()
    await background_delivery.reconcile_background_subtasks()
    assert "cancelled" in world.entry(world.parent.session_id).message


async def test_parent_stop_suppresses_wake(world: World):
    task = await world.launch()
    await background_delivery.cancel_background_subtasks(
        world.user_id, world.parent.session_id, task.launch_turn_id
    )
    await world.finish_parent()
    await background_delivery.reconcile_background_subtasks()
    assert len(world.published.await_args_list) == 1
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.wake_suppressed and stored.followup_state == "waiting"
    await stream_registry.mark_session_completed(
        task.child_session_id, turn_id=task.child_turn_id
    )
    await background_delivery.reconcile_background_subtasks()
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "suppressed"


async def test_scope_change_and_other_owner_cannot_receive_or_claim(world: World):
    task = await world.launch()
    assert not await background_task_db.get_background_task(task.id, "someone-else")
    assert not await background_task_db.claim_background_turn(
        task.id, "someone-else", task.child_session_id, task.child_turn_id, "child"
    )
    await world.finish_parent()
    await world.finish_child(task)
    await db.execute_raw_with_schema(
        'UPDATE {schema_prefix}"ChatSession" SET metadata=metadata || $2::jsonb WHERE id=$1',
        world.parent.session_id,
        '{"llm_auth_provider":"codex"}',
    )
    await background_delivery.reconcile_background_subtasks()
    assert len(world.published.await_args_list) == 1
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "suppressed"


@pytest.mark.parametrize("credential_only", [False, True])
async def test_connection_change_posts_one_notice_without_running_a_followup(
    world: World, credential_only: bool
):
    if credential_only:
        assert await model.update_session_llm_route(
            world.parent.session_id, world.user_id, "codex", "original-connection"
        )
        parent = await model.get_chat_session(world.parent.session_id, world.user_id)
        assert parent
        world.parent = parent
    task = await world.launch()
    assert await model.update_session_llm_route(
        world.parent.session_id, world.user_id, "codex", "new-connection"
    )
    await world.finish_child(task)
    cached = await model.get_chat_session(world.parent.session_id, world.user_id)
    assert cached and not any(m.id == task.wake_id for m in cached.messages)

    await background_delivery.reconcile_background_subtasks()
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "waiting"
    assert not await DbMessage.prisma().find_unique(where={"id": task.wake_id})
    await world.finish_parent()
    await background_delivery.reconcile_background_subtasks()
    await background_delivery.reconcile_background_subtasks()

    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "suppressed"
    assert len(world.published.await_args_list) == 1
    refreshed = await model.get_chat_session(world.parent.session_id, world.user_id)
    assert refreshed
    notices = [m for m in refreshed.messages if m.id == task.wake_id]
    assert len(notices) == 1 and notices[0].role == "assistant"
    assert "connection changed" in (notices[0].content or "")
    assert f"/copilot?sessionId={task.child_session_id}" in (notices[0].content or "")
    assert "The answer is 42." not in (notices[0].content or "")
    assert await chat_db().get_chat_session_status(world.parent.session_id) == "idle"


@pytest.mark.parametrize("scope", ["organization_id", "team_id", "expert_id"])
async def test_changed_scope_does_not_receive_a_connection_notice(
    world: World, scope: str
):
    task = await world.launch()
    await world.finish_parent()
    await world.finish_child(task)
    context = task.context.model_copy(update={scope: str(uuid4())})
    await db.execute_raw_with_schema(
        'UPDATE {schema_prefix}"ChatBackgroundTask" SET context=$2::jsonb WHERE id=$1',
        task.id,
        context.model_dump_json(),
    )
    assert await model.update_session_llm_route(
        world.parent.session_id, world.user_id, "codex", "new-connection"
    )

    await background_delivery.reconcile_background_subtasks()

    assert not await DbMessage.prisma().find_unique(where={"id": task.wake_id})
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "suppressed"
    assert len(world.published.await_args_list) == 1


async def test_launch_retry_and_worker_redelivery_do_not_duplicate_work(world: World):
    first = await world.launch()
    repeated = await world.launch()
    assert (
        repeated.id == first.id and repeated.child_session_id == first.child_session_id
    )
    entry = world.entry(first.child_session_id)
    assert await claim_background_execution(entry)
    assert not await claim_background_execution(entry)


@pytest.mark.parametrize("followup", [False, True])
async def test_duplicate_worker_leaves_the_claimed_turn_running(
    world: World, monkeypatch: pytest.MonkeyPatch, followup: bool
):
    task = await world.launch()
    session_id = task.child_session_id
    if followup:
        await world.finish_parent()
        await world.finish_child(task)
        await background_delivery.reconcile_background_subtasks()
        session_id = task.parent_session_id
    entry = world.entry(session_id)
    assert await claim_background_execution(entry)
    worker = processor.CoPilotProcessor()
    worker.execution_loop = asyncio.get_running_loop()
    run_turn = AsyncMock()
    monkeypatch.setattr(worker, "_run_turn", run_turn)

    await asyncio.to_thread(worker.execute, entry, threading.Event(), MagicMock())

    active = await stream_registry.get_session(session_id)
    assert active and active.turn_id == entry.turn_id and active.status == "running"
    assert await chat_db().get_chat_session_status(session_id) == "running"
    run_turn.assert_not_awaited()


@pytest.mark.parametrize("outcome", ["cancelled", "timed_out", "suppressed"])
async def test_rejected_pickup_closes_with_its_terminal_reason(
    world: World, monkeypatch: pytest.MonkeyPatch, outcome: str
):
    task = await world.launch()
    session_id = task.child_session_id
    if outcome == "suppressed":
        await world.finish_parent()
        await world.finish_child(task)
        await background_delivery.reconcile_background_subtasks()
        session_id = task.parent_session_id
        await background_task_db.cancel_background_tasks(
            world.user_id, session_id, task.launch_turn_id
        )
    elif outcome == "cancelled":
        await background_task_db.cancel_background_tasks(world.user_id, session_id)
    else:
        await background_task_db.complete_background_task(
            task.id, world.user_id, "timed_out"
        )
    entry = world.entry(session_id)
    worker = processor.CoPilotProcessor()
    worker.execution_loop = asyncio.get_running_loop()
    run_turn = AsyncMock()
    monkeypatch.setattr(worker, "_run_turn", run_turn)

    await asyncio.to_thread(worker.execute, entry, threading.Event(), MagicMock())

    active = await stream_registry.get_session(session_id)
    assert active and active.status == "failed"
    assert await chat_db().get_chat_session_status(session_id) == "idle"
    events: asyncio.Queue[stream_registry.StreamEntry] = asyncio.Queue()
    await stream_registry._replay(await get_redis_async(), entry.turn_id, "0-0", events)
    chunks = [events.get_nowait()[1] for _ in range(events.qsize())]
    errors = [chunk.errorText for chunk in chunks if isinstance(chunk, StreamError)]
    assert errors == [
        (
            BACKGROUND_TIMEOUT_MESSAGE
            if outcome == "timed_out"
            else stream_registry.CANCELLED_MESSAGE
        )
    ]
    run_turn.assert_not_awaited()


@pytest.mark.parametrize("followup", [False, True])
@pytest.mark.parametrize(
    "state", ["active", "missing_meta", "newer", "newer_admitting", "other_owner"]
)
async def test_deleted_handoff_releases_only_its_own_turn(
    world: World, monkeypatch: pytest.MonkeyPatch, followup: bool, state: str
):
    task = await world.launch()
    await world.finish_parent()
    session_id = task.child_session_id
    deleted_session_id = task.parent_session_id
    if followup:
        await world.finish_child(task)
        await background_delivery.reconcile_background_subtasks()
        session_id = task.parent_session_id
        deleted_session_id = task.child_session_id
    entry = world.entry(session_id)
    assert await chat_db().delete_chat_session(deleted_session_id, world.user_id)
    assert not await background_task_db.get_background_task(task.id, world.user_id)
    expected_turn_id = entry.turn_id
    if state in ("newer", "newer_admitting"):
        await stream_registry.mark_session_completed(session_id, turn_id=entry.turn_id)
        if state == "newer":
            expected_turn_id = await utils.schedule_chat_turn(
                session_id=session_id, user_id=world.user_id, message="New work."
            )
            assert expected_turn_id
        else:
            await stream_registry.delete_session_meta(session_id, turn_id=entry.turn_id)
            assert (
                await chat_db().admit_chat_session_turn(
                    session_id=session_id,
                    user_id=world.user_id,
                    expect_status="idle",
                    capacity=10,
                )
                == "admitted"
            )
    elif state == "missing_meta":
        await stream_registry.delete_session_meta(session_id, turn_id=entry.turn_id)
    elif state == "other_owner":
        entry = entry.model_copy(update={"user_id": "someone-else"})
    worker = processor.CoPilotProcessor()
    worker.execution_loop = asyncio.get_running_loop()
    run_turn = AsyncMock()
    monkeypatch.setattr(worker, "_run_turn", run_turn)

    for _ in range(2):
        await asyncio.to_thread(worker.execute, entry, threading.Event(), MagicMock())

    protected = state in ("newer", "newer_admitting", "other_owner")
    assert await chat_db().get_chat_session_status(session_id) == (
        "running" if protected else "idle"
    )
    active = await stream_registry.get_session(session_id)
    if state in ("missing_meta", "newer_admitting"):
        assert active is None
    else:
        assert active and active.turn_id == expected_turn_id
        assert active.status == ("running" if protected else "failed")
    if state == "active":
        events: asyncio.Queue[stream_registry.StreamEntry] = asyncio.Queue()
        await stream_registry._replay(
            await get_redis_async(), entry.turn_id, "0-0", events
        )
        chunks = [events.get_nowait()[1] for _ in range(events.qsize())]
        assert [c.errorText for c in chunks if isinstance(c, StreamError)] == [
            stream_registry.CANCELLED_MESSAGE
        ]
    run_turn.assert_not_awaited()


@pytest.mark.parametrize("followup", [False, True])
@pytest.mark.parametrize("reason", ["deadline", "budget"])
async def test_claimed_admission_failure_closes_with_its_terminal_reason(
    world: World, monkeypatch: pytest.MonkeyPatch, followup: bool, reason: str
):
    task = await world.launch()
    session_id = task.child_session_id
    if followup:
        await world.finish_parent()
        await world.finish_child(task)
        await background_delivery.reconcile_background_subtasks()
        session_id = task.parent_session_id
    entry = world.entry(session_id)
    error = (
        TimeoutError(BACKGROUND_TIMEOUT_MESSAGE)
        if reason == "deadline"
        else TreeRefusal("This task's budget has been used up.")
    )
    if reason == "deadline":
        monkeypatch.setattr(background_execution, "background_timeout", lambda _: 0)
    else:
        monkeypatch.setattr(
            background_execution, "readmit_turn", AsyncMock(side_effect=error)
        )
    worker = processor.CoPilotProcessor()
    worker.execution_loop = asyncio.get_running_loop()
    run_turn = AsyncMock()
    monkeypatch.setattr(worker, "_run_turn", run_turn)

    with pytest.raises(type(error)):
        await asyncio.to_thread(worker.execute, entry, threading.Event(), MagicMock())

    active = await stream_registry.get_session(session_id)
    assert active and active.status == "failed"
    assert await chat_db().get_chat_session_status(session_id) == "idle"
    events: asyncio.Queue[stream_registry.StreamEntry] = asyncio.Queue()
    await stream_registry._replay(await get_redis_async(), entry.turn_id, "0-0", events)
    chunks = [events.get_nowait()[1] for _ in range(events.qsize())]
    assert [c.errorText for c in chunks if isinstance(c, StreamError)] == [str(error)]
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored
    if followup:
        assert stored.followup_started_at and stored.followup_state == "finished"
    else:
        assert stored.started_at and stored.outcome == (
            "timed_out" if reason == "deadline" else "failed"
        )
    run_turn.assert_not_awaited()


async def test_worker_to_durable_handoff_to_parent_reply(
    world: World, monkeypatch: pytest.MonkeyPatch
):
    task = await world.launch()
    await world.finish_parent()
    generated: list[str] = []

    async def scripted_model(**kwargs: Any):
        session_id = kwargs["session_id"]
        generated.append(session_id)
        text = (
            "The answer is 42."
            if session_id == task.child_session_id
            else "Your background subtask found 42."
        )
        if session_id == task.parent_session_id:
            # The chat carries on under the launching turn's own envelope, with
            # its tools, and reads the result from the subtask's chat.
            assert kwargs["envelope"] == task.context.envelope
            assert kwargs["envelope"].permits("run_sub_session")
            assert task.child_session_id in kwargs["message"]
            assert "The answer is 42." not in kwargs["message"]
        await append_and_save_message(
            session_id, ChatMessage(role="assistant", content=text)
        )
        yield StreamTextDelta(id=str(uuid4()), delta=text)

    monkeypatch.setattr(processor, "ChatConfig", lambda: MagicMock(test_mode=True))
    monkeypatch.setattr(processor, "stream_chat_completion_dummy", scripted_model)
    monkeypatch.setattr(processor, "refusal_for", AsyncMock(return_value=None))
    monkeypatch.setattr(
        processor, "close_browser_daemon", AsyncMock(return_value=False)
    )
    worker = processor.CoPilotProcessor()
    log = CoPilotLogMetadata(logging.getLogger("background-test"))
    await worker._execute_async(
        world.entry(task.child_session_id), threading.Event(), MagicMock(), log
    )
    await asyncio.gather(
        background_delivery.reconcile_background_subtasks(),
        background_delivery.reconcile_background_subtasks(),
    )
    await worker._execute_async(
        world.entry(task.parent_session_id), threading.Event(), MagicMock(), log
    )
    await background_delivery.reconcile_background_subtasks()
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "finished"
    assert generated == [task.child_session_id, task.parent_session_id]
    replies = await DbMessage.prisma().find_many(
        where={"sessionId": task.parent_session_id, "role": "assistant"}
    )
    assert [reply.content for reply in replies] == ["Your background subtask found 42."]


@pytest.mark.parametrize(
    "completion",
    [
        "matching",
        "expired",
        "wrong_turn",
        "wrong_owner",
        "failed",
        "missing",
        "cancelled",
        "unstarted",
    ],
)
async def test_recovery_preserves_only_a_matching_success_after_a_completion_write_fails(
    world: World, monkeypatch: pytest.MonkeyPatch, completion: str
):
    task = await world.launch()
    await world.finish_parent()

    async def scripted_model(**_kwargs: Any):
        await append_and_save_message(
            task.child_session_id,
            ChatMessage(role="assistant", content="The answer is 42."),
        )
        yield StreamTextDelta(id=str(uuid4()), delta="The answer is 42.")

    monkeypatch.setattr(processor, "ChatConfig", lambda: MagicMock(test_mode=True))
    engine = MagicMock(side_effect=scripted_model)
    monkeypatch.setattr(processor, "stream_chat_completion_dummy", engine)
    monkeypatch.setattr(processor, "refusal_for", AsyncMock(return_value=None))
    monkeypatch.setattr(
        processor, "close_browser_daemon", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        processor,
        "finish_background_execution",
        AsyncMock(side_effect=ConnectionError("Completion write unavailable")),
    )

    assert await processor.CoPilotProcessor()._execute_async(
        world.entry(task.child_session_id),
        threading.Event(),
        MagicMock(),
        CoPilotLogMetadata(logging.getLogger("background-test")),
    )
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.outcome is None
    assert await chat_db().get_chat_session_status(task.child_session_id) == "idle"
    active = await stream_registry.get_session(task.child_session_id)
    assert active and active.status == "completed"
    redis = await get_redis_async()
    key = stream_registry.get_session_meta_key(task.child_session_id)
    if completion == "expired":
        await db.execute_raw_with_schema(
            'UPDATE {schema_prefix}"ChatBackgroundTask" SET deadline=$2::timestamptz WHERE id=$1',
            task.id,
            datetime.now(UTC) - timedelta(seconds=1),
        )
    elif completion == "wrong_turn":
        await redis.hset(key, "turn_id", str(uuid4()))
    elif completion == "wrong_owner":
        await redis.hset(key, "user_id", str(uuid4()))
    elif completion == "failed":
        await redis.hset(key, "status", "failed")
    elif completion == "missing":
        await redis.delete(key)
    elif completion == "cancelled":
        await background_task_db.cancel_background_tasks(
            world.user_id, task.child_session_id
        )
    elif completion == "unstarted":
        await db.execute_raw_with_schema(
            'UPDATE {schema_prefix}"ChatBackgroundTask" SET "startedAt"=NULL WHERE id=$1',
            task.id,
        )

    await background_delivery.reconcile_background_subtasks()
    await background_delivery.reconcile_background_subtasks()

    expected = (
        "completed"
        if completion in ("matching", "expired")
        else "cancelled" if completion == "cancelled" else "failed"
    )
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.outcome == expected
    wake = world.entry(world.parent.session_id)
    assert expected in wake.message
    engine.assert_called_once()
    assert len(world.published.await_args_list) == 2


async def test_completed_work_persists_even_when_cleanup_crosses_the_deadline(
    world: World, monkeypatch: pytest.MonkeyPatch
):
    task = await world.launch()
    finish = processor.finish_background_execution

    async def slow_finish(entry: CoPilotExecutionEntry, error: str | None):
        await asyncio.sleep(5.1)
        await finish(entry, error)

    async def scripted_model(**_kwargs: Any):
        await append_and_save_message(
            task.child_session_id,
            ChatMessage(role="assistant", content="Finished before the deadline."),
        )
        yield StreamTextDelta(id=str(uuid4()), delta="Finished before the deadline.")

    monkeypatch.setattr(processor, "background_timeout", lambda _: 5.0)
    monkeypatch.setattr(processor, "finish_background_execution", slow_finish)
    monkeypatch.setattr(processor, "ChatConfig", lambda: MagicMock(test_mode=True))
    monkeypatch.setattr(processor, "stream_chat_completion_dummy", scripted_model)
    monkeypatch.setattr(processor, "refusal_for", AsyncMock(return_value=None))
    monkeypatch.setattr(
        processor, "close_browser_daemon", AsyncMock(return_value=False)
    )
    worker = processor.CoPilotProcessor()
    worker.execution_loop = asyncio.get_running_loop()

    await asyncio.wait_for(
        asyncio.to_thread(
            worker.execute,
            world.entry(task.child_session_id),
            threading.Event(),
            MagicMock(),
        ),
        timeout=20,
    )

    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.outcome == "completed"
    active = await stream_registry.get_session(task.child_session_id)
    assert active and active.status == "completed"
    assert await chat_db().get_chat_session_status(task.child_session_id) == "idle"


@pytest.mark.parametrize("with_pending_message", [False, True])
async def test_stopping_a_queued_followup_reports_cancellation_once(
    world: World, monkeypatch: pytest.MonkeyPatch, with_pending_message: bool
):
    task = await world.launch()
    await world.finish_child(task)
    await world.finish_parent()
    monkeypatch.setattr(background_delivery, "dispatch_next_for_user", AsyncMock())
    await background_delivery.reconcile_background_subtasks()
    session = await model.get_chat_session(task.parent_session_id, world.user_id)
    assert session
    if with_pending_message:
        assert await persist_pending_as_user_rows(
            session,
            None,
            [PendingMessage(content="Keep the update short.")],
            log_prefix="queued-background-stop-test",
        )

    response = await cancel_session_task(task.parent_session_id, world.user_id)
    assert response.cancelled and response.reason == "dequeued"
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "finished"
    assert await chat_db().get_chat_session_status(task.parent_session_id) == "idle"
    await background_delivery.reconcile_background_subtasks()
    await background_delivery.reconcile_background_subtasks()

    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "finished"
    notices = await DbMessage.prisma().find_many(
        where={"sessionId": task.parent_session_id, "id": task.wake_id + "-notice"}
    )
    assert len(notices) == 1
    assert (notices[0].content or "").startswith(
        "Otto's background follow-up was cancelled."
    )
    assert len(world.published.await_args_list) == 1
    session = await model.get_chat_session(task.parent_session_id, world.user_id)
    assert session and any(m.content == notices[0].content for m in session.messages)


@pytest.mark.parametrize("boundary", ["other_owner", "promoted"])
async def test_queued_followup_stop_preserves_other_owners_and_promoted_turns(
    world: World, monkeypatch: pytest.MonkeyPatch, boundary: str
):
    task = await world.launch()
    await world.finish_child(task)
    await world.finish_parent()
    monkeypatch.setattr(background_delivery, "dispatch_next_for_user", AsyncMock())
    await background_delivery.reconcile_background_subtasks()
    user_id = world.user_id
    if boundary == "other_owner":
        user_id = str(uuid4())
    else:
        assert await turn_queue.dispatch_next_for_user(world.user_id)
    assert not await background_task_db.cancel_queued_background_followup(
        user_id, task.parent_session_id
    )
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "queued"
    assert await chat_db().get_chat_session_status(task.parent_session_id) == (
        "queued" if boundary == "other_owner" else "running"
    )
    assert not await DbMessage.prisma().find_unique(
        where={"id": task.wake_id + "-notice"}
    )


async def test_queued_followup_stop_rolls_back_when_notice_cannot_be_saved(
    world: World, monkeypatch: pytest.MonkeyPatch
):
    task = await world.launch()
    await world.finish_child(task)
    await world.finish_parent()
    monkeypatch.setattr(background_delivery, "dispatch_next_for_user", AsyncMock())
    await background_delivery.reconcile_background_subtasks()
    with monkeypatch.context() as failing_notice:
        failing_notice.setattr(
            background_task_db,
            "_append",
            AsyncMock(side_effect=RuntimeError("notice write failed")),
        )
        with pytest.raises(RuntimeError, match="notice write failed"):
            await cancel_session_task(task.parent_session_id, world.user_id)
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "queued"
    assert await chat_db().get_chat_session_status(task.parent_session_id) == "queued"
    assert not await DbMessage.prisma().find_unique(
        where={"id": task.wake_id + "-notice"}
    )
    assert (await cancel_session_task(task.parent_session_id, world.user_id)).cancelled


async def test_stopping_a_running_followup_reports_cancellation_once(
    world: World, monkeypatch: pytest.MonkeyPatch
):
    task = await world.launch()
    await world.finish_parent()
    await world.finish_child(task)
    await background_delivery.reconcile_background_subtasks()
    started = asyncio.Event()
    cancel = threading.Event()

    async def waiting_model(**_kwargs: Any):
        assert await model.get_chat_session(task.parent_session_id, world.user_id)
        started.set()
        await asyncio.sleep(30)
        yield StreamTextDelta(id="late", delta="Should have stopped")

    async def deliver_cancel(session_id: str):
        assert session_id == task.parent_session_id
        cancel.set()

    monkeypatch.setattr(
        "backend.api.features.chat.routes.enqueue_cancel_task", deliver_cancel
    )
    monkeypatch.setattr(processor, "ChatConfig", lambda: MagicMock(test_mode=True))
    monkeypatch.setattr(processor, "stream_chat_completion_dummy", waiting_model)
    monkeypatch.setattr(processor, "refusal_for", AsyncMock(return_value=None))
    monkeypatch.setattr(
        processor, "close_browser_daemon", AsyncMock(return_value=False)
    )
    worker = processor.CoPilotProcessor()
    worker.execution_loop = asyncio.get_running_loop()
    entry = world.entry(task.parent_session_id)
    running = asyncio.create_task(
        asyncio.to_thread(worker.execute, entry, cancel, MagicMock())
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        response = await cancel_session_task(task.parent_session_id, world.user_id)
        assert response.cancelled
    finally:
        cancel.set()
        await asyncio.wait_for(running, timeout=10)

    await finish_background_execution(entry, stream_registry.CANCELLED_MESSAGE)
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "finished"
    messages = await DbMessage.prisma().find_many(
        where={"sessionId": task.parent_session_id}
    )
    notices = [m for m in messages if m.id == task.wake_id + "-notice"]
    assert len(notices) == 1 and notices[0].role == "assistant"
    assert (notices[0].content or "").startswith(
        "Otto's background follow-up was cancelled."
    )
    assert task.child_session_id in (notices[0].content or "")

    session = await model.get_chat_session(task.parent_session_id, world.user_id)
    assert session and any(m.content == notices[0].content for m in session.messages)


@pytest.mark.parametrize(
    "error, notice",
    [
        (
            stream_registry.CANCELLED_MESSAGE,
            "Otto's background follow-up was cancelled.",
        ),
        (
            BACKGROUND_TIMEOUT_MESSAGE,
            "Otto couldn't complete the background follow-up within its limits.",
        ),
        (
            "Provider request failed.",
            "Otto couldn't complete the background follow-up.",
        ),
        ("Follow-up interrupted", "Otto couldn't complete the background follow-up."),
    ],
)
async def test_failed_followup_notice_matches_its_reason_once(
    world: World, error: str, notice: str
):
    task = await world.launch()
    await world.finish_child(task)
    await world.finish_parent()
    await background_delivery.reconcile_background_subtasks()
    entry = world.entry(task.parent_session_id)
    assert await claim_background_execution(entry)
    assert await model.get_chat_session(task.parent_session_id, world.user_id)

    await finish_background_execution(entry, error)
    await finish_background_execution(entry, error)

    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "finished"
    messages = await DbMessage.prisma().find_many(
        where={"sessionId": task.parent_session_id, "id": task.wake_id + "-notice"}
    )
    assert len(messages) == 1
    assert messages[0].content == (
        f"{notice} [Open the subtask](/copilot?sessionId={task.child_session_id}) for details."
    )
    session = await model.get_chat_session(task.parent_session_id, world.user_id)
    assert session and any(m.content == messages[0].content for m in session.messages)


async def test_stop_endpoint_cancels_an_accepted_queued_subtask(
    world: World, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(background_subtasks, "dispatch_next_for_user", AsyncMock())
    task = await world.launch()
    response = await cancel_session_task(task.child_session_id, world.user_id)
    assert response.cancelled
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.outcome == "cancelled"
    assert await chat_db().get_chat_session_status(task.child_session_id) == "idle"


async def test_cancelled_queued_child_gives_its_node_back(
    world: World, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(background_subtasks, "dispatch_next_for_user", AsyncMock())
    ledger = await get_tree_ledger()
    await ledger.open(
        world.envelope.tree_id,
        ceiling_microdollars=1_000_000,
        max_nodes=2,
        initial_nodes=1,
    )
    task = await world.launch()
    assert not world.published.await_count
    with pytest.raises(TreeRefusal, match="lifetime budget"):
        await world.launch("second-while-queued")
    response = await cancel_session_task(task.child_session_id, world.user_id)
    assert response.cancelled

    replacement = await world.launch("replacement")
    assert replacement.id != task.id
    await world.finish_parent()
    await background_delivery.reconcile_background_subtasks()
    wake = world.entry(task.parent_session_id)
    assert "cancelled" in wake.message


@pytest.mark.parametrize("dispatch_source", ["none", "scheduled", "chat", "direct"])
async def test_parent_stop_without_redis_cancels_children_and_suppresses_wake(
    world: World,
    dispatch_source: str,
):
    task = await world.launch()
    if dispatch_source != "none":
        with pytest.raises(turn_queue.SessionNotIdle):
            if dispatch_source == "chat":
                await utils.schedule_chat_turn(
                    session_id=world.parent.session_id,
                    user_id=world.user_id,
                    message="A request that lost the idle-slot race",
                )
            elif dispatch_source == "direct":
                await utils.dispatch_turn(
                    TurnSlot(world.user_id, world.parent.session_id),
                    session_id=world.parent.session_id,
                    user_id=world.user_id,
                    turn_id=str(uuid4()),
                    message="An unowned dispatch",
                )
            else:
                await utils.schedule_turn(
                    session_id=world.parent.session_id,
                    user_id=world.user_id,
                    turn_id=str(uuid4()),
                    message="A request that lost the idle-slot race",
                )
    await db.execute_raw_with_schema(
        'UPDATE {schema_prefix}"ChatSession" SET "updatedAt"=$2::timestamptz WHERE id=$1',
        world.parent.session_id,
        datetime.now(UTC) - timedelta(minutes=1),
    )
    await (await get_redis_async()).delete(
        stream_registry._get_session_meta_key(world.parent.session_id)
    )
    response = await cancel_session_task(world.parent.session_id, world.user_id)
    assert response.reason == "orphan_released"
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.outcome == "cancelled" and stored.wake_suppressed
    await stream_registry.mark_session_completed(
        task.child_session_id, turn_id=task.child_turn_id
    )
    await background_delivery.reconcile_background_subtasks()
    assert len(world.published.await_args_list) == 1
    assert await chat_db().get_chat_session_status(world.parent.session_id) == "idle"


async def test_orphan_stop_is_scoped_to_the_observed_turn_and_owner(world: World):
    old_task = await world.launch()
    old_parent = await chat_db().get_chat_session_metadata(world.parent.session_id)
    assert old_parent
    await world.finish_parent()
    set_execution_context(world.user_id, world.parent)
    new_turn_id = await utils.schedule_chat_turn(
        session_id=world.parent.session_id,
        user_id=world.user_id,
        message="A later request",
    )
    assert new_turn_id
    world.envelope = root_envelope(new_turn_id, session_id=world.parent.session_id)
    new_task = await world.launch("later-launch")
    assert (
        await background_task_db.cancel_orphaned_background_tasks(
            world.user_id, world.parent.session_id, old_parent.updated_at
        )
        is None
    )
    stale = datetime.now(UTC) - timedelta(minutes=1)
    await db.execute_raw_with_schema(
        'UPDATE {schema_prefix}"ChatSession" SET "updatedAt"=$2::timestamptz WHERE id=$1',
        world.parent.session_id,
        stale,
    )
    parent = await chat_db().get_chat_session_metadata(world.parent.session_id)
    assert parent
    assert (
        await background_task_db.cancel_orphaned_background_tasks(
            "another-owner", world.parent.session_id, parent.updated_at
        )
        is None
    )
    assert await chat_db().get_chat_session_status(world.parent.session_id) == "running"
    stored = await background_task_db.get_background_task(new_task.id, world.user_id)
    assert stored and stored.outcome is None
    await (await get_redis_async()).delete(
        stream_registry._get_session_meta_key(world.parent.session_id)
    )
    response = await cancel_session_task(world.parent.session_id, world.user_id)
    assert response.reason == "orphan_released"
    old = await background_task_db.get_background_task(old_task.id, world.user_id)
    new = await background_task_db.get_background_task(new_task.id, world.user_id)
    assert old and old.outcome is None and not old.wake_suppressed
    assert new and new.outcome == "cancelled" and new.wake_suppressed


async def test_late_cancellation_leaves_a_new_user_turn_queued(world: World):
    task = await world.launch()
    await append_and_save_message(
        task.child_session_id, ChatMessage(role="user", content="A separate request")
    )
    await chat_db().update_chat_session_status(
        session_id=task.child_session_id,
        user_id=world.user_id,
        expect_status="running",
        status="queued",
    )
    await background_delivery.cancel_background_subtasks(
        world.user_id, task.child_session_id
    )
    assert await chat_db().get_chat_session_status(task.child_session_id) == "queued"


@pytest.mark.parametrize(
    "failure_stage", ["before_meta", "after_meta", "replaced_meta"]
)
async def test_failed_dispatch_only_cleans_up_its_own_stream_metadata(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
):
    failed_turn_id = str(uuid4())
    replacement_turn_id = str(uuid4())
    slot = TurnSlot(world.user_id, world.parent.session_id)
    slot.admitted = True
    # The parent still records its launch turn; a dispatch follows an
    # admission, which clears that identity for the turn it then records.
    actual_db = chat_db()
    await actual_db.update_chat_session_status(
        session_id=world.parent.session_id, status="idle"
    )
    assert (
        await actual_db.admit_chat_session_turn(
            session_id=world.parent.session_id,
            user_id=world.user_id,
            expect_status="idle",
            capacity=5,
        )
        == "admitted"
    )
    if failure_stage == "before_meta":
        monkeypatch.setattr(
            utils,
            "chat_db",
            lambda: MagicMock(record_chat_session_turn=AsyncMock(return_value=False)),
        )

    async def fail_publish(**_kwargs):
        if failure_stage == "replaced_meta":
            await stream_registry.create_session(
                session_id=world.parent.session_id,
                user_id=world.user_id,
                tool_call_id="replacement",
                tool_name="chat",
                turn_id=replacement_turn_id,
            )
        raise RuntimeError("publish failed")

    monkeypatch.setattr(utils, "enqueue_copilot_turn", fail_publish)
    with pytest.raises(RuntimeError):
        await utils.dispatch_turn(
            slot,
            session_id=world.parent.session_id,
            user_id=world.user_id,
            turn_id=failed_turn_id,
            message="Work",
            envelope=world.envelope,
        )
    active = await stream_registry.get_session(world.parent.session_id)
    if failure_stage == "after_meta":
        assert active is None
    else:
        expected = (
            replacement_turn_id
            if failure_stage == "replaced_meta"
            else world.envelope.tree_id
        )
        assert active and active.turn_id == expected


async def test_dispatch_cancelled_before_recording_cannot_release_a_successor(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
):
    await world.finish_parent()
    failed_turn_id = str(uuid4())
    winner: str | None = None
    actual_db = chat_db()

    async def interrupted_record(session_id: str, user_id: str, turn_id: str):
        nonlocal winner
        if turn_id != failed_turn_id:
            return await actual_db.record_chat_session_turn(
                session_id, user_id, turn_id
            )
        await actual_db.update_chat_session_status(
            session_id=session_id,
            user_id=user_id,
            status="idle",
            expect_status="running",
        )
        recorded = await actual_db.record_chat_session_turn(
            session_id, user_id, turn_id
        )
        assert not recorded
        winner = await utils.schedule_chat_turn(
            session_id=session_id,
            user_id=user_id,
            message="A newer request",
        )
        return recorded

    monkeypatch.setattr(
        utils, "chat_db", lambda: MagicMock(record_chat_session_turn=interrupted_record)
    )
    with pytest.raises(RuntimeError, match="no longer running"):
        await utils.schedule_turn(
            session_id=world.parent.session_id,
            user_id=world.user_id,
            turn_id=failed_turn_id,
            message="A cancelled request",
        )
    assert winner
    assert await actual_db.get_chat_session_status(world.parent.session_id) == "running"
    active = await stream_registry.get_session(world.parent.session_id)
    assert active and active.turn_id == winner


@pytest.mark.parametrize("stream_lost", [False, True])
async def test_recovery_during_promotion_preserves_the_unstarted_followup(
    world: World, monkeypatch: pytest.MonkeyPatch, stream_lost: bool
):
    task = await world.launch()
    await world.finish_child(task)
    await world.finish_parent()
    claim = turn_queue.claim_queued_session
    recovered = False

    async def recover_after_claim(session, *, sub_work):
        nonlocal recovered
        result = await claim(session, sub_work=sub_work)
        if result == "admitted" and session.session_id == task.parent_session_id:
            if stream_lost:
                await stream_registry.delete_session_meta(
                    task.parent_session_id, turn_id=task.launch_turn_id
                )
            await background_delivery.reconcile_background_subtasks()
            recovered = True
        return result

    monkeypatch.setattr(turn_queue, "claim_queued_session", recover_after_claim)
    await background_delivery.reconcile_background_subtasks()

    assert recovered
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "queued"
    wake = world.entry(task.parent_session_id)
    assert wake.turn_id == task.wake_id
    assert await claim_background_execution(wake)
    assert not await claim_background_execution(wake)
    await finish_background_execution(wake, None)
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "finished"


@pytest_asyncio.fixture(loop_scope="session")
async def world(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[World]:
    connected = db.is_connected()
    if not connected:
        await db.connect()
    user_id = str(uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"{user_id}@example.invalid"}
    )
    published = AsyncMock()
    monkeypatch.setattr(utils, "enqueue_copilot_turn", published)
    monkeypatch.setattr(
        model, "_resolve_new_session_autopilot_mode", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        background_subtasks, "is_feature_enabled", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        background_subtasks,
        "capture_cost_context",
        AsyncMock(return_value=TrialCostContext(user_id=user_id, trial_id=None)),
    )
    monkeypatch.setattr(background_delivery, "enqueue_cancel_task", AsyncMock())
    monkeypatch.setattr(
        tree, "resolve_root_ceiling_microdollars", AsyncMock(return_value=1_000_000)
    )
    monkeypatch.setattr(turn_queue, "turn_refusal", AsyncMock(return_value=None))
    parent = await create_chat_session(user_id, dry_run=False)
    set_execution_context(user_id, parent)
    turn_id = await utils.schedule_chat_turn(
        session_id=parent.session_id,
        user_id=user_id,
        message="Research this in the background",
    )
    assert turn_id
    published.reset_mock()
    try:
        yield World(
            user_id=user_id,
            parent=parent,
            envelope=root_envelope(turn_id, session_id=parent.session_id),
            published=published,
        )
    finally:
        await User.prisma().delete(where={"id": user_id})
        if not connected:
            await db.disconnect()


async def test_session_state_reports_the_outcome_and_a_pending_continuation(
    world: World,
):
    task = await world.launch()
    parent_id, child_id = world.parent.session_id, task.child_session_id
    state = background_task_db.get_session_background_state

    assert await state(child_id, world.user_id) == BackgroundSessionState()
    assert await state(parent_id, world.user_id) == BackgroundSessionState(pending=True)
    assert await state(child_id, str(uuid4())) == BackgroundSessionState()

    await world.finish_parent()
    await world.finish_child(task)
    await background_delivery.reconcile_background_subtasks()
    assert (await state(child_id, world.user_id)).outcome == "completed"
    assert (await state(parent_id, world.user_id)).pending

    entry = world.entry(parent_id)
    assert await claim_background_execution(entry)
    await finish_background_execution(entry, None)
    assert await state(parent_id, world.user_id) == BackgroundSessionState()


@pytest.mark.parametrize("completion", ["matching", "missing", "wrong_turn"])
async def test_followup_recovery_keeps_a_delivered_continuation(
    world: World, completion: str
):
    task = await world.launch()
    await world.finish_child(task)
    await world.finish_parent()
    await background_delivery.reconcile_background_subtasks()
    entry = world.entry(task.parent_session_id)
    assert await claim_background_execution(entry)
    # The follow-up answered and closed its turn, but the write recording that
    # failed, which leaves the handoff queued for recovery.
    await append_and_save_message(
        task.parent_session_id,
        ChatMessage(role="assistant", content="The background result is 42."),
    )
    await stream_registry.mark_session_completed(
        task.parent_session_id, turn_id=task.wake_id
    )
    await chat_db().update_chat_session_status(
        session_id=task.parent_session_id, expect_status="running", status="idle"
    )
    redis = await get_redis_async()
    key = stream_registry.get_session_meta_key(task.parent_session_id)
    if completion == "missing":
        await redis.delete(key)
    elif completion == "wrong_turn":
        await redis.hset(key, "turn_id", str(uuid4()))

    await background_delivery.reconcile_background_subtasks()

    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "finished"
    notices = await DbMessage.prisma().find_many(
        where={"sessionId": task.parent_session_id, "id": task.wake_id + "-notice"}
    )
    assert len(notices) == (0 if completion == "matching" else 1)


async def test_a_dispatcher_that_lost_its_admission_cannot_record_its_turn(
    world: World,
):
    """Admission clears the recorded identity, so a dispatcher that stalled
    until orphan recovery freed its slot finds the successor's turn in place
    and refuses itself instead of overwriting it."""
    session_id, user_id = world.parent.session_id, world.user_id
    actual_db = chat_db()
    await actual_db.update_chat_session_status(session_id=session_id, status="idle")
    assert (
        await actual_db.admit_chat_session_turn(
            session_id=session_id, user_id=user_id, expect_status="idle", capacity=5
        )
        == "admitted"
    )

    assert await actual_db.record_chat_session_turn(session_id, user_id, "successor")
    assert not await actual_db.record_chat_session_turn(session_id, user_id, "stale")

    rows = await db.query_raw_with_schema(
        """SELECT metadata->>'_active_turn_id' AS turn
        FROM {schema_prefix}"ChatSession" WHERE id=$1""",
        session_id,
    )
    assert rows[0]["turn"] == "successor"
