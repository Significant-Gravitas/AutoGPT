"""Sub-sessions that wait for a slot, against the database: the real spawn,
queue, promotion and completion path, with only the executor's queue publish
stubbed."""

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from prisma.models import User
from pydantic import BaseModel, ConfigDict

from backend.api.features.chat.routes import cancel_session_task
from backend.copilot import active_turns, stream_registry, turn_queue
from backend.copilot.active_turns import acquire_turn_slot
from backend.copilot.context import set_execution_context
from backend.copilot.model import (
    CHAT_STATUS_IDLE,
    CHAT_STATUS_RUNNING,
    ChatSession,
    create_chat_session,
    get_chat_session,
)
from backend.copilot.permissions import ALL_TOOL_NAMES, CopilotPermissions
from backend.copilot.sdk.session_waiter import (
    SessionResult,
    run_copilot_turn_via_queue,
    wait_for_queued_session,
)
from backend.copilot.tools.get_sub_session_result import GetSubSessionResultTool
from backend.copilot.tree import (
    SpawnRequest,
    TurnEnvelope,
    get_tree_ledger,
    root_envelope,
)
from backend.data.db import prisma as db_client
from backend.data.db_accessors import chat_db
from backend.data.redis_client import get_redis_async
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_six_sub_sessions_from_one_turn_all_run_four_at_a_time(user: "_User"):
    """The 18:09Z shape: one turn starts six, three run and three wait; a
    message the user sends meanwhile runs at once and holds the fifth slot."""
    spawner = await user.running_chat()
    subs = await user.spawn(spawner, 6)
    assert [o for o, _ in subs] == ["running"] * 3 + ["queued_for_slot"] * 3

    typed = await user.chat()
    async with acquire_turn_slot(user.id, typed.session_id) as slot:
        assert slot.admitted
        await user.start(typed.session_id, slot)

    # The user's turn holds a slot, so the spawner's end frees none for sub-work.
    await user.end(spawner.session_id)
    assert user.dispatched_sessions() == [s for _, s in subs[:3]]
    await user.end(typed.session_id)
    for _, sub in subs:
        await user.end(sub)

    assert user.dispatched_sessions() == [s for _, s in subs]
    for _, sub in subs[3:]:
        promoted = user.dispatch_of(sub)
        assert promoted["envelope"].depth == 1
        assert promoted["envelope"].tree_id == spawner.tree_id
        meta = await stream_registry.get_session(sub)
        assert meta is not None
        assert (meta.tool_name, meta.tool_call_id) == (
            "run_sub_session",
            f"sub:{spawner.session_id}",
        )
    for _, sub in subs:
        assert await chat_db().get_chat_session_status(sub) == CHAT_STATUS_IDLE
    # The root plus one node per child: a promotion takes no second node.
    assert (await user.nodes(spawner.tree_id)) == 7


@pytest.mark.asyncio(loop_scope="session")
async def test_a_queued_sub_session_whose_tree_closed_is_refused_with_the_reason(
    user: "_User",
):
    spawner = await user.running_chat()
    subs = await user.spawn(spawner, 5)
    queued = [s for o, s in subs if o == "queued_for_slot"]
    assert len(queued) == 2
    await (await get_redis_async()).delete(
        (await get_tree_ledger()).key(spawner.tree_id)
    )

    await user.end(spawner.session_id)

    for sub in queued:
        assert await chat_db().get_chat_session_status(sub) == CHAT_STATUS_IDLE
        outcome, result = await wait_for_queued_session(
            session_id=sub, user_id=user.id, timeout=0
        )
        assert outcome == "refused"
        assert "tree has closed" in result.refusal
        assert sub not in user.dispatched_sessions()


@pytest.mark.asyncio(loop_scope="session")
async def test_cancelling_a_queued_sub_session_takes_it_out_and_returns_its_node(
    user: "_User",
):
    spawner = await user.running_chat()
    subs = await user.spawn(spawner, 4)
    (queued,) = [s for o, s in subs if o == "queued_for_slot"]
    tool = GetSubSessionResultTool()
    polled = await tool._execute(
        user.id, spawner.session, sub_session_id=queued, wait_if_running=0
    )
    assert polled.status == "queued"
    nodes = await user.nodes(spawner.tree_id)

    cancelled = await tool._execute(
        user.id, spawner.session, sub_session_id=queued, cancel=True
    )
    await user.end(spawner.session_id)

    assert cancelled.status == "cancelled"
    assert await chat_db().get_chat_session_status(queued) == CHAT_STATUS_IDLE
    assert await user.nodes(spawner.tree_id) == nodes - 1
    assert queued not in user.dispatched_sessions()


@pytest.mark.asyncio(loop_scope="session")
async def test_cancelling_a_queued_sub_session_over_http_returns_its_node(
    user: "_User",
):
    """The chat page's Stop button dequeues through the same path."""
    spawner = await user.running_chat()
    subs = await user.spawn(spawner, 4)
    (queued,) = [s for o, s in subs if o == "queued_for_slot"]
    nodes = await user.nodes(spawner.tree_id)

    response = await cancel_session_task(session_id=queued, user_id=user.id)
    await user.end(spawner.session_id)

    assert response.reason == "dequeued"
    assert await chat_db().get_chat_session_status(queued) == CHAT_STATUS_IDLE
    assert await user.nodes(spawner.tree_id) == nodes - 1
    assert queued not in user.dispatched_sessions()


@pytest.mark.asyncio(loop_scope="session")
async def test_the_inflight_cap_still_refuses_a_spawn(user: "_User"):
    spawner = await user.running_chat()
    with patch(
        "backend.copilot.executor.utils.get_inflight_turn_limit", return_value=6
    ):
        subs = await user.spawn(spawner, 6)

    assert [o for o, _ in subs] == (
        ["running"] * 3 + ["queued_for_slot"] * 2 + ["rejected_concurrent_turn_cap"]
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_tool_revoked_while_a_sub_session_waited_stays_revoked(
    user: "_User",
):
    spawner = await user.running_chat()
    subs = await user.spawn(spawner, 4)
    (queued,) = [s for o, s in subs if o == "queued_for_slot"]
    revoked = CopilotPermissions(tools=["web_fetch"], tools_exclude=True)

    with patch.object(turn_queue, "resolve_session_permissions", return_value=revoked):
        await user.end(spawner.session_id)

    promoted = user.dispatch_of(queued)
    granted = promoted["permissions"].effective_allowed_tools(ALL_TOOL_NAMES)
    assert "web_fetch" not in granted
    assert "web_search" in granted
    assert not promoted["envelope"].permits("web_fetch")


class _Spawner(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: ChatSession
    envelope: TurnEnvelope

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def tree_id(self) -> str:
        return self.envelope.tree_id


class _User:
    """A throwaway user whose turns publish to a stub instead of the executor."""

    def __init__(self, user_id: str, enqueued: AsyncMock) -> None:
        self.id = user_id
        self._enqueued = enqueued
        self._turns: dict[str, str] = {}

    async def chat(self) -> ChatSession:
        return await create_chat_session(self.id, dry_run=False)

    async def running_chat(self) -> _Spawner:
        session = await self.chat()
        turn_id = str(uuid.uuid4())
        assert await chat_db().update_chat_session_status(
            session_id=session.session_id,
            expect_status=CHAT_STATUS_IDLE,
            status=CHAT_STATUS_RUNNING,
            user_id=self.id,
        )
        await stream_registry.create_session(
            session.session_id, self.id, "chat_stream", "chat", turn_id
        )
        self._turns[session.session_id] = turn_id
        reloaded = await get_chat_session(session.session_id, self.id)
        assert reloaded is not None
        return _Spawner(
            session=reloaded,
            envelope=root_envelope(turn_id, session_id=session.session_id),
        )

    async def spawn(self, spawner: _Spawner, count: int) -> list[tuple[str, str]]:
        """``run_sub_session``'s spawn, ``count`` times, from inside the
        spawner's turn: each opens a delegated session and starts its turn."""

        async def in_the_spawners_turn() -> list[tuple[str, str]]:
            set_execution_context(self.id, spawner.session, envelope=spawner.envelope)
            spawned = []
            for _ in range(count):
                sub = await create_chat_session(
                    self.id,
                    dry_run=False,
                    origin="automation",
                    delegated_by_session_id=spawner.session_id,
                )
                outcome, _ = await run_copilot_turn_via_queue(
                    session_id=sub.session_id,
                    user_id=self.id,
                    message="test a stream",
                    timeout=0,
                    tool_call_id=f"sub:{spawner.session_id}",
                    tool_name="run_sub_session",
                    spawn=SpawnRequest(may_spawn=True, shares_memory=True),
                    allow_queue=False,
                )
                spawned.append((outcome, sub.session_id))
            return spawned

        return await asyncio.create_task(in_the_spawners_turn())

    async def start(self, session_id: str, slot) -> None:
        turn_id = str(uuid.uuid4())
        await stream_registry.create_session(
            session_id, self.id, "chat_stream", "chat", turn_id
        )
        self._turns[session_id] = turn_id
        slot.keep()

    async def end(self, session_id: str) -> None:
        """The executor's end of a turn, which promotes the next queued one."""
        turn_id = self._turns.get(session_id) or self.dispatch_of(session_id)["turn_id"]
        await stream_registry.mark_session_completed(session_id, turn_id=turn_id)

    def dispatched_sessions(self) -> list[str]:
        return [c.kwargs["session_id"] for c in self._enqueued.await_args_list]

    def dispatch_of(self, session_id: str) -> dict[str, Any]:
        (call,) = [
            c
            for c in self._enqueued.await_args_list
            if c.kwargs["session_id"] == session_id
        ]
        return call.kwargs

    async def nodes(self, tree_id: str) -> int:
        return (await (await get_tree_ledger()).snapshot(tree_id))["nodes"]


@pytest_asyncio.fixture
async def user(server: SpinTestServer) -> AsyncIterator[_User]:
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"queued-spawns-{user_id}@example.com"}
    )
    enqueued = AsyncMock()
    try:
        with (
            patch("backend.copilot.executor.utils.enqueue_copilot_turn", new=enqueued),
            patch.object(active_turns, "get_running_turn_limit", return_value=5),
            patch(
                "backend.copilot.executor.utils.get_inflight_turn_limit",
                return_value=15,
            ),
            patch.object(
                turn_queue, "is_user_paywalled", new=AsyncMock(return_value=False)
            ),
            patch.object(
                turn_queue,
                "get_global_rate_limits",
                new=AsyncMock(return_value=(1, 1, None)),
            ),
            patch.object(turn_queue, "check_rate_limit", new=AsyncMock()),
            patch(
                "backend.copilot.sdk.session_waiter.wait_for_session_result",
                new=AsyncMock(return_value=("running", SessionResult())),
            ),
        ):
            yield _User(user_id, enqueued)
    finally:
        await User.prisma().delete(where={"id": user_id})


@pytest_asyncio.fixture(autouse=True)
async def absorb_a_stale_event_loop(server: SpinTestServer):
    """An earlier test can leave the shared Prisma client bound to a closed
    loop; only the first query on the new loop fails. Spend it here."""
    try:
        await db_client.execute_raw("SELECT 1")
    except RuntimeError as error:
        if "Event loop is closed" not in str(error):
            raise
