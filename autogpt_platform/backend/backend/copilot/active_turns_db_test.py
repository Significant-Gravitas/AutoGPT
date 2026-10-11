"""The running cap against the database: what the spawn tools persist is
what the cap reads."""

import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from prisma.models import User

from backend.copilot import active_turns
from backend.copilot import db as copilot_db
from backend.copilot.active_turns import (
    ConcurrentTurnLimitError,
    TurnSlot,
    acquire_turn_slot,
)
from backend.copilot.model import (
    CHAT_STATUS_IDLE,
    CHAT_STATUS_QUEUED,
    CHAT_STATUS_RUNNING,
    create_chat_session,
)
from backend.copilot.sdk.session_waiter import SessionResult, run_copilot_turn_via_queue
from backend.data.db import prisma as db_client
from backend.data.db_accessors import chat_db
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_a_chat_fanning_out_six_sub_sessions_leaves_the_user_a_slot():
    """The 18:12Z shape: one chat's turn starts six sub-sessions, then the user
    sends a message in another chat."""
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"active-turns-{user_id}@example.com"}
    )
    try:
        spawner = await create_chat_session(user_id, dry_run=False)
        assert await chat_db().update_chat_session_status(
            session_id=spawner.session_id,
            expect_status=CHAT_STATUS_IDLE,
            status=CHAT_STATUS_RUNNING,
            user_id=user_id,
        )
        subs = [
            await create_chat_session(
                user_id,
                dry_run=False,
                origin="automation",
                delegated_by_session_id=spawner.session_id,
            )
            for _ in range(6)
        ]
        chat = await create_chat_session(user_id, dry_run=False)

        with (
            patch.object(active_turns, "get_running_turn_limit", return_value=5),
            patch(
                "backend.copilot.executor.utils.get_inflight_turn_limit",
                return_value=15,
            ),
            patch("backend.copilot.executor.utils.dispatch_turn", new=_keep_slot),
            patch(
                "backend.copilot.sdk.session_waiter.wait_for_session_result",
                new=AsyncMock(return_value=("running", SessionResult())),
            ),
        ):
            outcomes = [
                (
                    await run_copilot_turn_via_queue(
                        session_id=sub.session_id,
                        user_id=user_id,
                        message="test a stream",
                        timeout=0,
                        tool_call_id=f"sub:{spawner.session_id}",
                        tool_name="run_sub_session",
                        allow_queue=False,
                    )
                )[0]
                for sub in subs
            ]
            async with acquire_turn_slot(user_id, chat.session_id) as slot:
                assert slot.admitted

        assert outcomes == ["running"] * 3 + ["rejected_concurrent_turn_cap"] * 3
    finally:
        await User.prisma().delete(where={"id": user_id})


@pytest.mark.parametrize(
    "capacity, free",
    [
        # Six sub-sessions spawned at once with three running: one fits.
        (4, 1),
        # Six messages at once with three running: two fit.
        (5, 2),
    ],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_admits_fill_exactly_the_free_slots(capacity: int, free: int):
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"active-turns-{user_id}@example.com"}
    )
    try:
        for _ in range(3):
            running = await create_chat_session(user_id, dry_run=False)
            assert await chat_db().update_chat_session_status(
                session_id=running.session_id,
                expect_status=CHAT_STATUS_IDLE,
                status=CHAT_STATUS_RUNNING,
                user_id=user_id,
            )
        waiting = [await create_chat_session(user_id, dry_run=False) for _ in range(6)]

        admitted = await asyncio.gather(
            *(_admit(user_id, s.session_id, capacity) for s in waiting)
        )

        assert sum(admitted) == free
        assert (
            await chat_db().count_chat_sessions_by_status(
                user_id=user_id, status=CHAT_STATUS_RUNNING
            )
        ) == 3 + free
    finally:
        await User.prisma().delete(where={"id": user_id})


@pytest.mark.asyncio(loop_scope="session")
async def test_a_claim_on_a_row_that_moved_after_it_was_read_is_busy():
    """A cancel outside the lock can land between the read and the update."""
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"active-turns-{user_id}@example.com"}
    )
    try:
        session = await create_chat_session(user_id, dry_run=False)
        real = copilot_db.PrismaChatSession.prisma

        with patch.object(
            copilot_db.PrismaChatSession,
            "prisma",
            side_effect=lambda client=None: _ReadBeforeTheCancel(real(client)),
        ):
            admit = await copilot_db.admit_chat_session_turn(
                session_id=session.session_id,
                user_id=user_id,
                expect_status=CHAT_STATUS_QUEUED,
                capacity=5,
            )

        assert admit == "busy"
        assert (
            await chat_db().get_chat_session_status(session.session_id)
        ) == CHAT_STATUS_IDLE
    finally:
        await User.prisma().delete(where={"id": user_id})


class _ReadBeforeTheCancel:
    """The session's actions, with its read taken just before a cancel moved it
    from queued to idle."""

    def __init__(self, actions) -> None:
        self._actions = actions

    def __getattr__(self, name: str):
        return getattr(self._actions, name)

    async def find_unique(self, **kwargs):
        row = await self._actions.find_unique(**kwargs)
        return row.model_copy(update={"chatStatus": CHAT_STATUS_QUEUED})


async def _admit(user_id: str, session_id: str, capacity: int) -> bool:
    try:
        async with acquire_turn_slot(user_id, session_id, capacity=capacity) as slot:
            slot.keep()
            return slot.admitted
    except ConcurrentTurnLimitError:
        return False


async def _keep_slot(slot: TurnSlot, **_) -> None:
    slot.keep()


@pytest_asyncio.fixture(autouse=True)
async def absorb_a_stale_event_loop(server: SpinTestServer):
    """An earlier test can leave the shared Prisma client bound to a closed
    loop; only the first query on the new loop fails. Spend it here."""
    try:
        await db_client.execute_raw("SELECT 1")
    except RuntimeError as error:
        if "Event loop is closed" not in str(error):
            raise
