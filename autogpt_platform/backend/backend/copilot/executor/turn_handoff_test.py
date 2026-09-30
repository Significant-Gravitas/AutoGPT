"""A chat's next turn, started by the one ending, on one executor end to end.

One real ``CoPilotExecutor`` whose queue hands every turn straight back to it,
which is the worst case and what happened on prod: the ending turn started the
approval follow-up while it still sat in ``active_tasks``, the executor dropped
it as a duplicate, and the chat read as running a turn nobody ran. A script
stands in for the engine; the wake, the dispatch, the executor's intake and
done-callback, and the approved call's run are the real ones. Every coroutine
runs on the test's loop, where the Prisma client lives: the executor's threads
hand theirs over.
"""

import asyncio
import json
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.copilot import stream_registry
from backend.copilot.active_turns import acquire_turn_slot
from backend.copilot.executor.manager import CoPilotExecutor
from backend.copilot.executor.utils import CoPilotExecutionEntry, dispatch_turn
from backend.copilot.gate import held
from backend.copilot.gate.held_db_test import _answer, _hold, _new_session, _Post
from backend.copilot.model import get_chat_session
from backend.data.db_accessors import chat_db
from backend.data.redis_client import get_redis_async
from backend.data.redis_helpers import as_str

_MANAGER = "backend.copilot.executor.manager"


@pytest.fixture
def gate_on():
    with patch("backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=True)):
        yield


@pytest.fixture
def post_tool():
    tool = _Post()
    with patch("backend.copilot.tools.get_tool", return_value=tool):
        yield tool


@pytest.mark.asyncio(loop_scope="session")
async def test_cards_answered_during_a_turn_run_once_that_turn_has_left(
    setup_test_user, test_user_id, gate_on, post_tool
):
    session = await _new_session(test_user_id)

    async def turn(entry: CoPilotExecutionEntry) -> None:
        chat = await get_chat_session(entry.session_id, entry.user_id)
        assert chat is not None
        async for _status in held.resolve_answered(test_user_id, chat, _ignore):
            pass
        if entry.message == "post it":
            review_id = await _hold(chat, test_user_id, "approved mid-turn")
            await _answer(review_id, ReviewStatus.APPROVED)
        await stream_registry.mark_session_completed(
            entry.session_id, turn_id=entry.turn_id
        )

    async with _Pod.running(turn) as pod:
        await _start_turn(session.session_id, test_user_id, "post it")
        final = await _settled(pod, session.session_id, turns=2)

    assert pod.dropped == [], "the executor dropped the follow-up turn unrun"
    assert [t.message for t in pod.turns] == ["post it", held.WAKE_MESSAGE]
    assert post_tool.runs == [{"text": "approved mid-turn"}]
    assert (final.turn_id, final.status) == (pod.turns[1].turn_id, "completed")


@pytest.mark.asyncio(loop_scope="session")
async def test_a_turn_dropped_unrun_ends_and_a_redelivered_running_one_does_not(
    setup_test_user, test_user_id
):
    """A second turn can arrive while the first is still closing on this pod."""
    session = await _new_session(test_user_id)
    finish, closing, tail = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def turn(entry: CoPilotExecutionEntry) -> None:
        await finish.wait()
        await stream_registry.mark_session_completed(
            entry.session_id, turn_id=entry.turn_id
        )
        closing.set()
        await tail.wait()

    async with _Pod.running(turn) as pod:
        first = await _start_turn(session.session_id, test_user_id, "first")
        pod.deliver(pod.bodies[0])
        await asyncio.sleep(1)
        while_running = await stream_registry.get_session(session.session_id)
        finish.set()
        await closing.wait()
        second = await _start_turn(session.session_id, test_user_id, "second")
        chunks = await _chunks_until_finish(second)
        tail.set()

    assert pod.dropped == [first, second]
    assert while_running is not None
    assert (while_running.turn_id, while_running.status) == (first, "running")
    assert [c["type"] for c in chunks] == ["error", "finish"]
    assert await stream_registry.get_active_session(
        session.session_id, test_user_id
    ) == (None, "0-0")
    info = await chat_db().get_chat_session_metadata(session.session_id)
    assert info is not None and info.chat_status == "idle"


@pytest.mark.asyncio(loop_scope="session")
async def test_a_turn_the_executor_cannot_read_still_ends():
    """A newer pod's message, say: its ids still name the turn to close."""
    session_id, turn_id = str(uuid.uuid4()), str(uuid.uuid4())
    await stream_registry.create_session(session_id, None, "", "", turn_id=turn_id)
    body = CoPilotExecutionEntry(
        session_id=session_id, turn_id=turn_id, user_id=None, message="hi"
    ).model_dump(mode="json") | {"llm_auth_provider": "from-a-newer-pod"}

    async with _Pod.running(_never_runs) as pod:
        pod.deliver(json.dumps(body).encode())
        chunks = await _chunks_until_finish(turn_id)

    assert pod.turns == []
    assert [c["type"] for c in chunks] == ["error", "finish"]
    closed = await stream_registry.get_session(session_id)
    assert closed is not None and closed.status == "failed"


class _Pod:
    """One executor whose queue delivers every turn published back to itself."""

    def __init__(self, script: Callable[[CoPilotExecutionEntry], Awaitable[None]]):
        self.loop = asyncio.get_running_loop()
        self.script = script
        self.bodies: list[bytes] = []
        self.turns: list[CoPilotExecutionEntry] = []
        self.dropped: list[str] = []
        self.executor = CoPilotExecutor()
        self.executor._executor = ThreadPoolExecutor(max_workers=2)
        self.channel = SimpleNamespace(
            is_open=True,
            connection=SimpleNamespace(add_callback_threadsafe=lambda cb: cb()),
            basic_ack=lambda _tag: None,
            basic_nack=self._nack,
        )

    @classmethod
    @asynccontextmanager
    async def running(
        cls, script: Callable[[CoPilotExecutionEntry], Awaitable[None]]
    ) -> AsyncIterator["_Pod"]:
        pod = cls(script)
        # The executor's own loops (after a turn, closing a dropped one) run
        # on this test's loop instead of a fresh one per thread.
        on_loop = SimpleNamespace(
            run=pod._on_loop,
            sleep=asyncio.sleep,
            CancelledError=asyncio.CancelledError,
        )
        with (
            patch(f"{_MANAGER}.execute_copilot_turn", pod._run_turn),
            patch(f"{_MANAGER}.asyncio", on_loop),
            patch(
                "backend.util.clients.get_async_copilot_queue",
                AsyncMock(return_value=pod),
            ),
        ):
            try:
                yield pod
            finally:
                await pod._idle()
                pod.executor._executor.shutdown(wait=False)

    def deliver(self, body: bytes) -> None:
        self.bodies.append(body)
        self.executor._handle_run_message(
            self.channel,
            SimpleNamespace(delivery_tag=len(self.bodies)),
            None,
            body,
        )

    async def publish_message(self, routing_key: str, message: str, exchange: Any):
        self.deliver(message.encode())

    def _run_turn(self, entry: CoPilotExecutionEntry, *_args: Any) -> None:
        self.turns.append(entry)
        asyncio.run_coroutine_threadsafe(self.script(entry), self.loop).result(60)

    def _on_loop(self, coro: Awaitable[None]) -> None:
        asyncio.run_coroutine_threadsafe(coro, self.loop).result(60)

    async def _idle(self) -> None:
        # The executor's own threads finish on this loop, so wait, never join.
        for _ in range(300):
            busy = [
                t
                for t in threading.enumerate()
                if t.name.startswith(("after-turn-", "close-rejected-"))
            ]
            if not busy and not self.executor.active_tasks:
                return
            await asyncio.sleep(0.1)

    def _nack(self, tag: int, requeue: bool) -> None:
        if not requeue:
            ids = json.loads(self.bodies[tag - 1])
            self.dropped.append(ids["turn_id"])


async def _start_turn(session_id: str, user_id: str, message: str) -> str:
    turn_id = str(uuid.uuid4())
    async with acquire_turn_slot(user_id, session_id) as slot:
        await dispatch_turn(
            slot,
            session_id=session_id,
            user_id=user_id,
            turn_id=turn_id,
            message=message,
        )
    return turn_id


async def _settled(
    pod: _Pod, session_id: str, turns: int
) -> stream_registry.ActiveSession:
    """The chat once its ``turns``-th turn ended, or once the pod dropped one."""
    meta = None
    for _ in range(300):
        meta = await stream_registry.get_session(session_id)
        if pod.dropped or (
            len(pod.turns) == turns
            and meta is not None
            and meta.turn_id == pod.turns[-1].turn_id
            and meta.status != "running"
        ):
            break
        await asyncio.sleep(0.1)
    assert meta is not None
    return meta


async def _chunks_until_finish(turn_id: str) -> list[dict[str, Any]]:
    redis = await get_redis_async()
    chunks: list[dict[str, Any]] = []
    for _ in range(300):
        entries = await redis.xrange(stream_registry._get_turn_stream_key(turn_id))
        chunks = [json.loads(as_str(fields["data"]) or "") for _id, fields in entries]
        if chunks and chunks[-1]["type"] == "finish":
            break
        await asyncio.sleep(0.1)
    return chunks


def _ignore(_result: object) -> None:
    pass


async def _never_runs(_entry: CoPilotExecutionEntry) -> None:
    raise AssertionError("an unreadable turn must not run")
