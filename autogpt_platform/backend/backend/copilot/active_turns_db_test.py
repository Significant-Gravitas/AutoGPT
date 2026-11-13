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
from backend.data.db import (
    execute_raw_with_schema,
    prisma,
    query_raw_with_schema,
    transaction,
)
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

        assert outcomes == ["running"] * 3 + ["queued_for_slot"] * 3
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
async def test_a_claim_on_a_row_a_cancel_moves_mid_admit_is_busy():
    """A cancel outside the lock can move the row after the admit read it:
    the flip waits for the cancel and finds the row no longer queued."""
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"active-turns-{user_id}@example.com"}
    )
    try:
        session = await create_chat_session(user_id, dry_run=False)
        assert await chat_db().update_chat_session_status(
            session_id=session.session_id,
            expect_status=CHAT_STATUS_IDLE,
            status=CHAT_STATUS_QUEUED,
            user_id=user_id,
        )
        cancel_backend: asyncio.Queue[int] = asyncio.Queue(maxsize=1)
        cancel_may_commit = asyncio.Event()

        async def cancel() -> None:
            async with transaction() as tx:
                assert await copilot_db.PrismaChatSession.prisma(tx).update_many(
                    where={"id": session.session_id, "chatStatus": CHAT_STATUS_QUEUED},
                    data={"chatStatus": CHAT_STATUS_IDLE},
                )
                rows = await tx.query_raw("SELECT pg_backend_pid() AS pid")
                cancel_backend.put_nowait(rows[0]["pid"])
                await cancel_may_commit.wait()

        cancelling = asyncio.create_task(cancel())
        admitting = None
        try:
            blocker_pid = await asyncio.wait_for(cancel_backend.get(), timeout=10)
            admitting = asyncio.create_task(
                copilot_db.admit_chat_session_turn(
                    session_id=session.session_id,
                    user_id=user_id,
                    expect_status=CHAT_STATUS_QUEUED,
                    capacity=5,
                )
            )
            await _wait_for_database_lock(blocker_pid, "transactionid")
        finally:
            cancel_may_commit.set()
            await asyncio.gather(
                cancelling,
                *([admitting] if admitting is not None else []),
                return_exceptions=True,
            )
        await cancelling
        assert admitting is not None
        assert await admitting == "busy"
        assert (
            await chat_db().get_chat_session_status(session.session_id)
        ) == CHAT_STATUS_IDLE
    finally:
        await User.prisma().delete(where={"id": user_id})


@pytest.mark.asyncio(loop_scope="session")
async def test_an_admit_that_waited_on_the_lock_is_stamped_after_the_wait():
    """The stale-slot release takes a running row past its grace for an
    orphan, so an admit queued behind the user's lock stamps the time it
    flipped the row, not the time its transaction began."""
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"active-turns-{user_id}@example.com"}
    )
    try:
        session = await create_chat_session(user_id, dry_run=False)
        holder_backend: asyncio.Queue[int] = asyncio.Queue(maxsize=1)
        release = asyncio.Event()

        async def hold_the_users_lock() -> None:
            async with transaction() as tx:
                await tx.execute_raw(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    f"copilot-turns:{user_id}",
                )
                rows = await tx.query_raw("SELECT pg_backend_pid() AS pid")
                holder_backend.put_nowait(rows[0]["pid"])
                await release.wait()

        holder = asyncio.create_task(hold_the_users_lock())
        admitting = None
        try:
            blocker_pid = await asyncio.wait_for(holder_backend.get(), timeout=10)
            admitting = asyncio.create_task(
                copilot_db.admit_chat_session_turn(
                    session_id=session.session_id,
                    user_id=user_id,
                    expect_status=CHAT_STATUS_IDLE,
                    capacity=5,
                )
            )
            observed_ms = await _wait_for_database_lock(blocker_pid, "advisory")
        finally:
            release.set()
            await asyncio.gather(
                holder,
                *([admitting] if admitting is not None else []),
                return_exceptions=True,
            )
        await holder
        assert admitting is not None
        assert await admitting == "admitted"
        row = await copilot_db.PrismaChatSession.prisma().find_unique(
            where={"id": session.session_id}
        )
        assert row is not None
        assert round(row.updatedAt.timestamp() * 1000) >= observed_ms
    finally:
        await User.prisma().delete(where={"id": user_id})


@pytest.mark.parametrize(
    "status, capacity, admit",
    [
        (CHAT_STATUS_IDLE, 5, "admitted"),
        (CHAT_STATUS_RUNNING, 5, "busy"),
        (CHAT_STATUS_IDLE, 1, "full"),
    ],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_an_admit_flips_only_an_idle_session_under_the_cap(
    status: str, capacity: int, admit: str
):
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"active-turns-{user_id}@example.com"}
    )
    try:
        running = await create_chat_session(user_id, dry_run=False)
        assert await chat_db().update_chat_session_status(
            session_id=running.session_id,
            expect_status=CHAT_STATUS_IDLE,
            status=CHAT_STATUS_RUNNING,
            user_id=user_id,
        )
        session = await create_chat_session(user_id, dry_run=False)
        if status != CHAT_STATUS_IDLE:
            assert await chat_db().update_chat_session_status(
                session_id=session.session_id,
                expect_status=CHAT_STATUS_IDLE,
                status=status,
                user_id=user_id,
            )
        # The turn the session last recorded: an admit drops it so the next
        # dispatch can record its own; a refusal leaves a running turn's alone.
        await execute_raw_with_schema(
            """UPDATE {schema_prefix}"ChatSession"
            SET metadata = metadata || jsonb_build_object('_active_turn_id', $2::text)
            WHERE id = $1""",
            session.session_id,
            "previous-turn",
        )
        before = await copilot_db.PrismaChatSession.prisma().find_unique(
            where={"id": session.session_id}
        )
        assert before is not None
        before_turn, before_rest = await _recorded_turn(session.session_id)
        assert before_turn == "previous-turn"

        assert (
            await copilot_db.admit_chat_session_turn(
                session_id=session.session_id,
                user_id=user_id,
                expect_status=CHAT_STATUS_IDLE,
                capacity=capacity,
            )
            == admit
        )

        after = await copilot_db.PrismaChatSession.prisma().find_unique(
            where={"id": session.session_id}
        )
        assert after is not None
        after_turn, after_rest = await _recorded_turn(session.session_id)
        # The rest of the metadata is never touched.
        assert after_rest == before_rest
        if admit == "admitted":
            assert after.chatStatus == CHAT_STATUS_RUNNING
            # Every transition touches the row: updatedAt is its time.
            assert after.updatedAt > before.updatedAt
            assert after_turn is None
        else:
            assert after.chatStatus == status
            assert after.updatedAt == before.updatedAt
            assert after_turn == "previous-turn"
    finally:
        await User.prisma().delete(where={"id": user_id})


@pytest.mark.asyncio(loop_scope="session")
async def test_another_users_session_is_never_admitted():
    owner_id, other_id = str(uuid.uuid4()), str(uuid.uuid4())
    for user_id in (owner_id, other_id):
        await User.prisma().create(
            data={"id": user_id, "email": f"active-turns-{user_id}@example.com"}
        )
    try:
        session = await create_chat_session(owner_id, dry_run=False)

        assert (
            await copilot_db.admit_chat_session_turn(
                session_id=session.session_id,
                user_id=other_id,
                expect_status=CHAT_STATUS_IDLE,
                capacity=5,
            )
            == "busy"
        )
        assert (
            await chat_db().get_chat_session_status(session.session_id)
        ) == CHAT_STATUS_IDLE
    finally:
        for user_id in (owner_id, other_id):
            await User.prisma().delete(where={"id": user_id})


async def _wait_for_database_lock(blocker_pid: int, wait_event: str) -> float:
    """Observe the admit blocked by this test's holder and return DB time in ms.

    The holder owns a unique user's row or advisory lock, so its sole waiter
    is this test's admit. Each observation uses a new implicit transaction.
    Separate its start from the observation by more than the stored timestamp's
    millisecond precision, so a transaction-start stamp cannot pass by rounding.
    """
    async with asyncio.timeout(10):
        while True:
            rows = await prisma.query_raw(
                """SELECT pid,
                    floor(extract(epoch FROM clock_timestamp()) * 1000)::float8
                        AS observed_ms
                FROM pg_stat_activity
                WHERE datname = current_database() AND state = 'active'
                    AND wait_event_type = 'Lock' AND wait_event = $2::text
                    AND $1::int = ANY(pg_blocking_pids(pid))
                    AND xact_start < date_trunc('milliseconds', clock_timestamp())
                        - interval '1 millisecond'""",
                blocker_pid,
                wait_event,
            )
            if rows:
                assert len(rows) == 1, rows
                return rows[0]["observed_ms"]
            await asyncio.sleep(0.01)


async def _admit(user_id: str, session_id: str, capacity: int) -> bool:
    try:
        async with acquire_turn_slot(user_id, session_id, capacity=capacity) as slot:
            slot.keep()
            return slot.admitted
    except ConcurrentTurnLimitError:
        return False


async def _keep_slot(slot: TurnSlot, **_) -> None:
    slot.keep()


async def _recorded_turn(session_id: str) -> tuple[str | None, str]:
    """The session's recorded turn, and the rest of its metadata as text."""
    rows = await query_raw_with_schema(
        """SELECT metadata->>'_active_turn_id' AS turn,
            (metadata - '_active_turn_id')::text AS rest
        FROM {schema_prefix}"ChatSession" WHERE id = $1""",
        session_id,
    )
    return rows[0]["turn"], rows[0]["rest"]


@pytest_asyncio.fixture(autouse=True)
async def absorb_a_stale_event_loop(server: SpinTestServer):
    """An earlier test can leave the shared Prisma client bound to a closed
    loop; only the first query on the new loop fails. Spend it here."""
    try:
        await prisma.execute_raw("SELECT 1")
    except RuntimeError as error:
        if "Event loop is closed" not in str(error):
            raise
