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
from backend.copilot.background_execution import (
    claim_background_execution,
    finish_background_execution,
)
from backend.copilot.background_task import BackgroundTask
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
from backend.copilot.response_model import StreamTextDelta
from backend.copilot.tool_call_context import tool_call_context
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

    async def launch(self, call_id: str = "launch") -> BackgroundTask:
        set_execution_context(self.user_id, self.parent, envelope=self.envelope)
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
    turn_id = str(uuid4())
    await chat_db().update_chat_session_status(
        session_id=parent.session_id,
        expect_status="idle",
        status="running",
        user_id=user_id,
    )
    await stream_registry.create_session(
        parent.session_id, user_id, "chat_stream", "chat", turn_id
    )
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
