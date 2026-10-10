"""Queue promotion against the database: a queued turn starts from the
slot-free hook, which runs inside the turn that just ended."""

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from prisma.models import User

from backend.copilot import stream_registry, turn_queue
from backend.copilot.context import set_execution_context
from backend.copilot.gate import held
from backend.copilot.model import (
    CHAT_STATUS_IDLE,
    CHAT_STATUS_QUEUED,
    CHAT_STATUS_RUNNING,
    ChatSession,
    create_chat_session,
    get_chat_session,
)
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.tree import TurnEnvelope, admit_turn, get_tree_ledger
from backend.data.db import prisma as db_client
from backend.data.db_accessors import chat_db
from backend.util.test import SpinTestServer


@pytest.mark.parametrize("finished_depth", [1, 3])
@pytest.mark.asyncio(loop_scope="session")
async def test_a_promoted_message_is_a_root_not_the_finished_turns_child(
    finished_depth: int,
):
    finished = _envelope(finished_depth)
    promoted, nodes_before, nodes_after = await _promote_after(
        finished, message="the message that waited", message_metadata=None
    )

    assert promoted is not None, "the queued message was never dispatched"
    assert promoted.depth == 0
    assert promoted.tools is None
    assert promoted.tree_id != finished.tree_id
    assert nodes_after == nodes_before


@pytest.mark.asyncio(loop_scope="session")
async def test_a_queued_wake_starts_under_the_envelope_its_call_was_held_under():
    held_under = _envelope(1)
    ledger = await get_tree_ledger()
    await ledger.open(held_under.tree_id, ceiling_microdollars=1_000_000, max_nodes=10)
    revoked = CopilotPermissions(tools=["web_fetch"], tools_exclude=True)

    # A tool revoked while the wake waited stays revoked.
    with patch.object(turn_queue, "resolve_session_permissions", return_value=revoked):
        promoted, _, _ = await _promote_after(
            _envelope(2),
            message=held.WAKE_MESSAGE,
            message_metadata={held._WAKE_KEY: True},
            envelope=held_under,
        )

    assert promoted is not None
    assert (promoted.tree_id, promoted.depth) == (held_under.tree_id, 1)
    assert not promoted.permits("web_fetch")
    # The call's turn was counted when it ran; its wake takes no second node.
    assert (await ledger.snapshot(held_under.tree_id))["nodes"] == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_a_queued_wake_with_no_recorded_envelope_is_not_started():
    """Not derived from the turn that happens to free the slot."""
    promoted, _, _ = await _promote_after(
        _envelope(1),
        message=held.WAKE_MESSAGE,
        message_metadata={held._WAKE_KEY: True},
        expect_refusal=turn_queue.UNRECORDED_WAKE,
    )

    assert promoted is None


@pytest.mark.asyncio(loop_scope="session")
async def test_a_queued_wake_whose_envelope_no_longer_parses_is_not_started():
    """A stored record from before a schema change is as good as none."""
    promoted, _, _ = await _promote_after(
        _envelope(1),
        message=held.WAKE_MESSAGE,
        message_metadata={held._WAKE_KEY: True, "envelope": {"depth": "deep"}},
        expect_refusal=turn_queue.UNRECORDED_WAKE,
    )

    assert promoted is None


@pytest.mark.asyncio(loop_scope="session")
async def test_queued_sub_work_is_not_promoted_into_the_users_last_slot():
    """Four still run after the turn ends: the fifth slot stays the user's."""
    promoted, _, _ = await _promote_after(
        _envelope(1), message="sub-work", message_metadata=None, sub_work=True
    )

    assert promoted is None


async def _promote_after(
    finished: TurnEnvelope,
    *,
    message: str,
    message_metadata: dict[str, Any] | None,
    envelope: TurnEnvelope | None = None,
    expect_refusal: str | None = None,
    sub_work: bool = False,
) -> tuple[TurnEnvelope | None, int, int]:
    """Queue a turn behind a full cap, then end a turn carrying ``finished``.

    Returns the promoted turn's envelope and the finished tree's node count
    before and after."""
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"turn-queue-{user_id}@example.com"}
    )
    try:
        sessions = [await create_chat_session(user_id, dry_run=False) for _ in range(5)]
        for session in sessions:
            assert await chat_db().update_chat_session_status(
                session_id=session.session_id,
                expect_status=CHAT_STATUS_IDLE,
                status=CHAT_STATUS_RUNNING,
                user_id=user_id,
            )
        ending, ending_turn = sessions[0], str(uuid.uuid4())
        await stream_registry.create_session(
            ending.session_id, user_id, "chat_stream", "chat", ending_turn
        )
        waiting = await create_chat_session(
            user_id,
            dry_run=False,
            delegated_by_session_id=sessions[1].session_id if sub_work else None,
        )
        await turn_queue.try_enqueue_turn(
            user_id=user_id,
            inflight_cap=15,
            session_id=waiting.session_id,
            message=message,
            message_metadata=message_metadata,
            envelope=envelope,
        )
        assert (
            await chat_db().get_chat_session_status(waiting.session_id)
        ) == CHAT_STATUS_QUEUED

        await admit_turn(finished, user_id=user_id)
        ledger = await get_tree_ledger()
        nodes_before = (await ledger.snapshot(finished.tree_id))["nodes"]

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
            await asyncio.create_task(_end_turn(user_id, finished, ending, ending_turn))

        promoted = [
            call.kwargs["envelope"]
            for call in enqueued.await_args_list
            if call.kwargs["session_id"] == waiting.session_id
        ]
        nodes_after = (await ledger.snapshot(finished.tree_id))["nodes"]
        if expect_refusal is not None:
            assert (
                await chat_db().get_chat_session_status(waiting.session_id)
            ) == CHAT_STATUS_IDLE
            closed = await get_chat_session(waiting.session_id, user_id)
            assert closed is not None
            assert turn_queue.queued_turn_refusal(closed) == expect_refusal
        return (promoted[0] if promoted else None), nodes_before, nodes_after
    finally:
        await User.prisma().delete(where={"id": user_id})


def _envelope(depth: int) -> TurnEnvelope:
    """A sub-session's turn, deep in a tree of its own."""
    return TurnEnvelope(
        tree_id=str(uuid.uuid4()),
        depth=depth,
        tools=frozenset({"run_sub_session", "connect_integration", "web_fetch"}),
    )


async def _end_turn(
    user_id: str, envelope: TurnEnvelope, session: ChatSession, turn_id: str
) -> None:
    set_execution_context(user_id, session, envelope=envelope)
    await stream_registry.mark_session_completed(session.session_id, turn_id=turn_id)


@pytest_asyncio.fixture(autouse=True)
async def absorb_a_stale_event_loop(server: SpinTestServer):
    """An earlier test can leave the shared Prisma client bound to a closed
    loop; only the first query on the new loop fails. Spend it here."""
    try:
        await db_client.execute_raw("SELECT 1")
    except RuntimeError as error:
        if "Event loop is closed" not in str(error):
            raise
