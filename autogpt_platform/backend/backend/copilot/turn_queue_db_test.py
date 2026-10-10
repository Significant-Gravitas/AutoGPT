"""Queue promotion against the database: a queued turn starts from the
slot-free hook, which runs inside the turn that just ended."""

import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from prisma.models import User

from backend.copilot import stream_registry, turn_queue
from backend.copilot.context import set_execution_context
from backend.copilot.model import (
    CHAT_STATUS_IDLE,
    CHAT_STATUS_QUEUED,
    CHAT_STATUS_RUNNING,
    create_chat_session,
)
from backend.copilot.tree import TurnEnvelope, admit_turn, get_tree_ledger
from backend.data.db import prisma as db_client
from backend.data.db_accessors import chat_db
from backend.util.test import SpinTestServer


@pytest.mark.parametrize("finished_depth", [1, 3])
@pytest.mark.asyncio(loop_scope="session")
async def test_a_promoted_message_is_a_root_not_the_finished_turns_child(
    finished_depth: int,
):
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"turn-queue-{user_id}@example.com"}
    )
    try:
        sessions = [await create_chat_session(user_id, dry_run=False) for _ in range(5)]
        for session in sessions:
            await _flip(user_id, session.session_id, CHAT_STATUS_RUNNING)
        finished, finished_turn = sessions[0], str(uuid.uuid4())
        await stream_registry.create_session(
            finished.session_id, user_id, "chat_stream", "chat", finished_turn
        )
        waiting = await create_chat_session(user_id, dry_run=False)
        await turn_queue.try_enqueue_turn(
            user_id=user_id,
            inflight_cap=15,
            session_id=waiting.session_id,
            message="the message that waited",
        )
        assert (
            await chat_db().get_chat_session_status(waiting.session_id)
        ) == CHAT_STATUS_QUEUED

        # The turn that ends is a sub-session's, deep in another tree.
        finished_envelope = TurnEnvelope(
            tree_id=str(uuid.uuid4()),
            depth=finished_depth,
            tools=frozenset({"run_sub_session", "connect_integration", "web_fetch"}),
        )
        await admit_turn(finished_envelope, user_id=user_id)
        ledger = await get_tree_ledger()
        nodes_before = (await ledger.snapshot(finished_envelope.tree_id))["nodes"]

        enqueued = AsyncMock()
        with (
            patch("backend.copilot.executor.utils.enqueue_copilot_turn", new=enqueued),
            patch.object(
                turn_queue, "is_user_paywalled", new=AsyncMock(return_value=False)
            ),
            patch.object(
                turn_queue,
                "get_global_rate_limits",
                new=AsyncMock(return_value=(1, 1, None)),
            ),
            patch.object(turn_queue, "check_rate_limit", new=AsyncMock()),
        ):
            # The executor sets the envelope inside the engine's generator and
            # completes the turn in the same task, so the hook inherits it.
            await asyncio.create_task(
                _end_turn(user_id, finished_envelope, finished, finished_turn)
            )

        promoted = [
            call.kwargs
            for call in enqueued.await_args_list
            if call.kwargs["session_id"] == waiting.session_id
        ]
        assert len(promoted) == 1, "the queued message was never dispatched"
        assert promoted[0]["envelope"].depth == 0
        assert promoted[0]["envelope"].tools is None
        assert promoted[0]["envelope"].tree_id != finished_envelope.tree_id
        assert (await ledger.snapshot(finished_envelope.tree_id))[
            "nodes"
        ] == nodes_before
    finally:
        await User.prisma().delete(where={"id": user_id})


async def _end_turn(
    user_id: str, envelope: TurnEnvelope, finished, finished_turn: str
) -> None:
    set_execution_context(user_id, finished, envelope=envelope)
    await stream_registry.mark_session_completed(
        finished.session_id, turn_id=finished_turn
    )


async def _flip(user_id: str, session_id: str, status: str) -> None:
    assert await chat_db().update_chat_session_status(
        session_id=session_id,
        expect_status=CHAT_STATUS_IDLE,
        status=status,
        user_id=user_id,
    )


@pytest_asyncio.fixture(autouse=True)
async def absorb_a_stale_event_loop(server: SpinTestServer):
    """An earlier test can leave the shared Prisma client bound to a closed
    loop; only the first query on the new loop fails. Spend it here."""
    try:
        await db_client.execute_raw("SELECT 1")
    except RuntimeError as error:
        if "Event loop is closed" not in str(error):
            raise
