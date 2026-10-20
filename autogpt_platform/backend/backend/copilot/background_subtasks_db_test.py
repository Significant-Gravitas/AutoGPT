"""Real Postgres/Redis handoffs, with provider calls and Rabbit publish replaced."""

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
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
    DELIVERY_SECONDS,
    BackgroundTask,
)
from backend.copilot.context import set_execution_context
from backend.copilot.executor import processor, utils
from backend.copilot.executor.utils import CoPilotExecutionEntry, CoPilotLogMetadata
from backend.copilot.llm_attribution import set_turn_trace_id
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    append_and_save_message,
    create_chat_session,
)
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
        self, call_id: str = "launch", permissions: CopilotPermissions | None = None
    ) -> BackgroundTask:
        set_execution_context(
            self.user_id, self.parent, permissions=permissions, envelope=self.envelope
        )
        set_turn_trace_id(self.envelope.tree_id)
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
    assert "The answer is 42." in wake.message
    assert wake.envelope and wake.envelope.tree_id == task.context.envelope.tree_id
    assert wake.envelope.depth > 0 and wake.envelope.tainted
    assert wake.envelope.tools == frozenset()
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


@pytest.mark.parametrize("blocks_exclude", [False, True])
async def test_background_child_keeps_block_restrictions_after_queuing(
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
    assert not entry.envelope.permits("run_sub_session")


async def test_spent_tree_refuses_followup_without_a_fresh_budget(world: World):
    task = await world.launch()
    await world.finish_parent()
    await world.finish_child(task)
    await (await get_tree_ledger()).charge(task.context.envelope.tree_id, 1_000_000)
    await background_delivery.reconcile_background_subtasks()
    assert len(world.published.await_args_list) == 1
    messages = await DbMessage.prisma().find_many(
        where={"sessionId": world.parent.session_id}
    )
    assert any(
        m.role == "assistant" and "budget" in (m.content or "").lower()
        for m in messages
    )


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
    assert stored.result == "The answer is 42."


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
    with pytest.raises(TimeoutError):
        await processor.CoPilotProcessor()._execute_async(
            entry,
            threading.Event(),
            MagicMock(),
            CoPilotLogMetadata(logging.getLogger("background-timeout-test")),
        )
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.outcome == "timed_out"


async def test_child_stop_reports_cancellation_and_parent_stop_suppresses_wake(
    world: World,
):
    task = await world.launch()
    await background_delivery.cancel_background_subtasks(
        world.user_id, task.child_session_id
    )
    await world.finish_parent()
    await background_delivery.reconcile_background_subtasks()
    assert '"outcome": "cancelled"' in world.entry(world.parent.session_id).message


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


async def test_deadline_is_recovered_even_without_worker_completion(world: World):
    task = await world.launch()
    await world.finish_parent()
    await db.execute_raw_with_schema(
        'UPDATE {schema_prefix}"ChatBackgroundTask" SET deadline=$2::timestamptz WHERE id=$1',
        task.id,
        datetime.now(UTC) - timedelta(seconds=1),
    )
    await background_delivery.reconcile_background_subtasks()
    assert '"outcome": "timed_out"' in world.entry(world.parent.session_id).message


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
            assert kwargs["envelope"].tools == frozenset()
            assert "The answer is 42." in kwargs["message"]
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


async def test_expired_worker_without_redis_meta_releases_its_slot(world: World):
    task = await world.launch()
    await world.finish_parent()
    deadline = datetime.now(UTC) - timedelta(seconds=1)
    await db.execute_raw_with_schema(
        'UPDATE {schema_prefix}"ChatBackgroundTask" SET deadline=$2::timestamptz WHERE id=$1',
        task.id,
        deadline,
    )
    await db.execute_raw_with_schema(
        'UPDATE {schema_prefix}"ChatSession" SET "updatedAt"=$2::timestamptz WHERE id=$1',
        task.child_session_id,
        deadline - timedelta(seconds=1),
    )
    await (await get_redis_async()).delete(
        stream_registry._get_session_meta_key(task.child_session_id)
    )
    await background_delivery.reconcile_background_subtasks()
    assert await chat_db().get_chat_session_status(task.child_session_id) == "idle"
    assert '"outcome": "timed_out"' in world.entry(task.parent_session_id).message


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


async def test_busy_parent_past_delivery_window_gets_one_notice_when_idle(world: World):
    task = await world.launch()
    await world.finish_child(task)
    await db.execute_raw_with_schema(
        'UPDATE {schema_prefix}"ChatBackgroundTask" SET deadline=$2::timestamptz WHERE id=$1',
        task.id,
        datetime.now(UTC) - timedelta(seconds=DELIVERY_SECONDS + 1),
    )
    await background_delivery.reconcile_background_subtasks()
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "waiting"
    await world.finish_parent()
    await background_delivery.reconcile_background_subtasks()
    await background_delivery.reconcile_background_subtasks()
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "suppressed"
    messages = await DbMessage.prisma().find_many(
        where={"sessionId": world.parent.session_id}
    )
    notices = [m for m in messages if m.role == "assistant"]
    assert len(notices) == 1
    assert "follow-up window expired" in (notices[0].content or "")
    assert task.child_session_id in (notices[0].content or "")
    assert len(world.published.await_args_list) == 1


@pytest.mark.parametrize("running", [False, True])
async def test_expired_followup_is_recovered_once(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    running: bool,
):
    task = await world.launch()
    await world.finish_child(task)
    await world.finish_parent()
    if not running:
        monkeypatch.setattr(background_delivery, "dispatch_next_for_user", AsyncMock())
    await background_delivery.reconcile_background_subtasks()
    if running:
        assert await claim_background_execution(world.entry(world.parent.session_id))
    pending = await chat_db().get_latest_user_message_in_session(
        world.parent.session_id
    )
    assert pending and pending.metadata
    envelope = TurnEnvelope.model_validate(pending.metadata["envelope"])
    assert envelope.deadline_at
    clock = MagicMock(wraps=datetime)
    clock.now.return_value = envelope.deadline_at + timedelta(seconds=10)
    monkeypatch.setattr(background_delivery, "datetime", clock)
    await background_delivery.reconcile_background_subtasks()
    await background_delivery.reconcile_background_subtasks()
    await background_delivery.reconcile_background_subtasks()
    stored = await background_task_db.get_background_task(task.id, world.user_id)
    assert stored and stored.followup_state == "finished"
    assert await chat_db().get_chat_session_status(world.parent.session_id) == "idle"
    messages = await DbMessage.prisma().find_many(
        where={"sessionId": world.parent.session_id}
    )
    notices = [m for m in messages if m.id == task.wake_id + "-notice"]
    assert len(notices) == 1 and notices[0].role == "assistant"
    assert "within its limits" in (notices[0].content or "")
    assert task.child_session_id in (notices[0].content or "")
    assert len(world.published.await_args_list) == (2 if running else 1)


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
