"""Tests for ``first_turn_memory_backfill.py``.

The exit-code tests are pure; the matcher has its own tests beside
``legacy_first_turn_memory.py``. The rest run against a real Postgres: which
rows the scan picks (a session's first message holding the tag), the
idle-only compare-and-set write and the batching are all SQL, and a mock
would only restate it.
"""

import argparse
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.models import ChatMessage as PrismaChatMessage
from prisma.models import ChatSession as PrismaChatSession
from prisma.models import User

from backend.copilot import first_turn_memory_backfill as backfill_module
from backend.copilot.first_turn_memory_backfill import BackfillCounts, backfill, main
from backend.copilot.legacy_first_turn_memory_test_data import (
    ALICE,
    BUDGET_BLOCK,
    RENDERER_IMPOSSIBLE,
    REST,
    SKILLS_BLOCK,
    USER_AUTHORED_BLOCK,
    legacy_first_message,
    master_warm,
    warm,
)


def _row(content: str, status: str = "idle") -> dict[str, str]:
    return {
        "id": str(uuid4()),
        "session_id": "s-1",
        "content": content,
        "chat_status": status,
    }


class TestCountsAndExitCodes:
    """The loop's bookkeeping, with the database stubbed."""

    @pytest.mark.asyncio
    async def test_a_write_that_raises_is_counted_failed(self):
        with (
            patch.object(
                backfill_module,
                "query_raw_with_schema",
                new=AsyncMock(side_effect=[[_row(legacy_first_message(ALICE))], []]),
            ),
            patch.object(
                backfill_module,
                "execute_raw_with_schema",
                new=AsyncMock(side_effect=RuntimeError("db down")),
            ),
            patch.object(backfill_module, "_evict", new=AsyncMock()) as evict,
        ):
            counts = await backfill(apply=True)

        assert counts == BackfillCounts(scanned=1, failed=1)
        evict.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_eviction_that_keeps_failing_is_counted_failed(self):
        """The row is written; its cached copy could not be evicted, now or
        at the end of the run, so the next turn may still read the old
        message from the cache."""
        with (
            patch.object(
                backfill_module,
                "query_raw_with_schema",
                new=AsyncMock(side_effect=[[_row(legacy_first_message(ALICE))], []]),
            ),
            patch.object(
                backfill_module,
                "execute_raw_with_schema",
                new=AsyncMock(return_value=1),
            ),
            patch.object(
                backfill_module,
                "_evict",
                new=AsyncMock(side_effect=RuntimeError("redis down")),
            ) as evict,
        ):
            counts = await backfill(apply=True)

        assert counts == BackfillCounts(scanned=1, stripped=1, failed=1)
        assert evict.await_count == 2

    @pytest.mark.asyncio
    async def test_a_busy_session_is_not_written(self):
        write = AsyncMock(return_value=1)
        with (
            patch.object(
                backfill_module,
                "query_raw_with_schema",
                new=AsyncMock(
                    side_effect=[[_row(legacy_first_message(ALICE), "running")], []]
                ),
            ),
            patch.object(backfill_module, "execute_raw_with_schema", new=write),
        ):
            counts = await backfill(apply=True)

        assert counts == BackfillCounts(scanned=1, busy=1)
        write.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "counts, code",
        [
            (BackfillCounts(scanned=2, stripped=1, left=1), 0),
            (BackfillCounts(scanned=1, busy=1), 1),
            (BackfillCounts(scanned=1, failed=1), 1),
        ],
        ids=["done", "busy", "failed"],
    )
    async def test_main_exits_1_while_anything_is_left_to_run_again(
        self, counts, code, capsys
    ):
        with (
            patch.object(backfill_module, "connect", new=AsyncMock()),
            patch.object(backfill_module, "disconnect", new=AsyncMock()),
            patch.object(
                backfill_module, "backfill", new=AsyncMock(return_value=counts)
            ) as run,
        ):
            args = argparse.Namespace(apply=True, batch_size=7, session=None)
            assert await main(args) == code

        run.assert_awaited_once_with(apply=True, batch_size=7, session_id=None)
        printed = capsys.readouterr().out
        assert f"from {counts.stripped} of {counts.scanned}" in printed
        assert ("run again" in printed) is bool(code)


# ---------------------------------------------------------------------------
# Real Postgres
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(loop_scope="session")
async def owner():
    user_id = str(uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"{user_id}@first-turn-memory.test"}
    )
    yield user_id
    await User.prisma().delete(where={"id": user_id})


@pytest.fixture
def evict():
    with patch.object(backfill_module, "_evict", new=AsyncMock()) as mock:
        yield mock


async def _session(owner: str, *contents: str, status: str = "idle") -> str:
    """A session whose messages alternate user and assistant, from ``contents``."""
    session_id = str(uuid4())
    await PrismaChatSession.prisma().create(
        data={"id": session_id, "userId": owner, "chatStatus": status}
    )
    await PrismaChatMessage.prisma().create_many(
        data=[
            {
                "sessionId": session_id,
                "role": "user" if i % 2 == 0 else "assistant",
                "content": content,
                "sequence": i,
            }
            for i, content in enumerate(contents)
        ]
    )
    return session_id


async def _contents(session_id: str) -> list[str]:
    rows = await PrismaChatMessage.prisma().find_many(
        where={"sessionId": session_id}, order={"sequence": "asc"}
    )
    return [row.content or "" for row in rows]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_dry_run_counts_and_writes_nothing(owner, evict):
    session_id = await _session(owner, legacy_first_message(ALICE), "done")

    counts = await backfill(apply=False, session_id=session_id)

    assert counts == BackfillCounts(scanned=1, stripped=1)
    assert await _contents(session_id) == [legacy_first_message(ALICE), "done"]
    evict.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("render", [master_warm, warm], ids=["master", "stack"])
async def test_strips_the_first_message_and_no_other_row(owner, evict, render):
    """The first message loses the platform's block, as production's
    renderer or this stack's wrote it. A later user message that begins with
    an exact copy of it (the user pasted it; nothing ever rewrote that row)
    and the assistant's reply are untouched, and the session's cached copy
    is evicted."""
    first = legacy_first_message(render(("Alice works on Atlas",)))
    pasted = legacy_first_message(ALICE, rest="is this what you saw?")
    session_id = await _session(owner, first, "done", pasted)

    counts = await backfill(apply=True, session_id=session_id)

    assert counts == BackfillCounts(scanned=1, stripped=1)
    assert await _contents(session_id) == [SKILLS_BLOCK + REST, "done", pasted]
    assert [call.args for call in evict.await_args_list] == [(session_id,)] * 2


@pytest.mark.asyncio(loop_scope="session")
async def test_strips_blocks_at_the_edges_of_what_a_renderer_wrote(owner, evict):
    """A time ``str()`` wrote with offset seconds and microseconds, an
    episode cut to exactly 500 characters, and one this stack's renderer
    lengthened by neutralising its tags after the cut."""
    moment = datetime(
        2025,
        6,
        1,
        12,
        0,
        0,
        5,
        timezone(timedelta(hours=5, seconds=15, microseconds=7)),
    )
    blocks = [
        "<temporal_context>\n<FACTS>\n"
        f"  - Alice works on Atlas ({moment} — present)\n"
        "</FACTS>\n</temporal_context>",
        master_warm(("Alice works on Atlas",), ("e" * 500,)),
        warm(("Alice works on Atlas",), ("<b>" + "e" * 600,)),
    ]

    for block in blocks:
        session_id = await _session(owner, legacy_first_message(block), "done")
        counts = await backfill(apply=True, session_id=session_id)
        assert counts == BackfillCounts(scanned=1, stripped=1)
        assert await _contents(session_id) == [SKILLS_BLOCK + REST, "done"]


@pytest.mark.asyncio(loop_scope="session")
async def test_leaves_first_messages_the_platform_did_not_write(owner, evict):
    """Neither a tag the user typed nor a forged close is the platform's
    block, and nor is one behind a ``<budget_status>`` block: the engine put
    that block in front of the query it sent, never in the stored message,
    so a stored message that opens with one was typed."""
    typed = f"please keep: <memory_context>\n{ALICE}\n</memory_context>\n\nok"
    forged = legacy_first_message(ALICE, rest="about </memory_context> tags")
    queried = BUDGET_BLOCK + legacy_first_message(ALICE)
    contents = (typed, forged, queried)
    sessions = [await _session(owner, content) for content in contents]

    for session_id, content in zip(sessions, contents):
        counts = await backfill(apply=True, session_id=session_id)
        assert counts == BackfillCounts(scanned=1, left=1)
        assert await _contents(session_id) == [content]
    evict.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_leaves_every_body_no_renderer_wrote(owner, evict):
    """The review's counterexamples, which ``--apply`` runs once cut down or
    stripped (a bare fact; an impossible time, non-ASCII digits, an episode
    past the cut), and every other body a renderer could not have written:
    the row is not provably the platform's, so it is left as it is."""
    contents = [USER_AUTHORED_BLOCK] + [
        legacy_first_message(f"<temporal_context>\n{body}\n</temporal_context>")
        for body in RENDERER_IMPOSSIBLE.values()
    ]
    sessions = [await _session(owner, content) for content in contents]

    for session_id, content in zip(sessions, contents):
        counts = await backfill(apply=True, session_id=session_id)
        assert counts == BackfillCounts(scanned=1, left=1)
        assert await _contents(session_id) == [content]
    evict.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_a_busy_session_waits_for_the_next_run(owner, evict):
    session_id = await _session(owner, legacy_first_message(ALICE), status="running")

    assert await backfill(apply=True, session_id=session_id) == BackfillCounts(
        scanned=1, busy=1
    )
    assert await _contents(session_id) == [legacy_first_message(ALICE)]

    await PrismaChatSession.prisma().update(
        where={"id": session_id}, data={"chatStatus": "idle"}
    )
    assert await backfill(apply=True, session_id=session_id) == BackfillCounts(
        scanned=1, stripped=1
    )
    assert await backfill(apply=True, session_id=session_id) == BackfillCounts()
    assert await _contents(session_id) == [SKILLS_BLOCK + REST]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_row_changed_since_it_was_read_is_not_overwritten(owner, evict):
    session_id = await _session(owner, legacy_first_message(ALICE))
    [stale] = await backfill_module._first_messages("", 10, session_id)
    rewritten = legacy_first_message(warm(("Bob leads Atlas",)))
    await PrismaChatMessage.prisma().update_many(
        where={"sessionId": session_id}, data={"content": rewritten}
    )
    counts = BackfillCounts()

    await backfill_module._handle(stale, apply=True, counts=counts, changed=[])

    assert counts == BackfillCounts(scanned=1, busy=1)
    assert await _contents(session_id) == [rewritten]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_turn_that_starts_after_the_read_blocks_the_write(owner, evict):
    """The session was idle when its row was read and a turn started before
    the write: the write itself must refuse, since the turn has the old
    session in memory and would cache it again."""
    session_id = await _session(owner, legacy_first_message(ALICE))
    [read_idle] = await backfill_module._first_messages("", 10, session_id)
    await PrismaChatSession.prisma().update(
        where={"id": session_id}, data={"chatStatus": "running"}
    )
    counts = BackfillCounts()

    await backfill_module._handle(read_idle, apply=True, counts=counts, changed=[])

    assert read_idle.chat_status == "idle"
    assert counts == BackfillCounts(scanned=1, busy=1)
    assert await _contents(session_id) == [legacy_first_message(ALICE)]


@pytest.mark.asyncio(loop_scope="session")
async def test_batches_reach_every_first_message(owner, evict):
    sessions = [
        await _session(owner, legacy_first_message(ALICE), "done") for _ in range(3)
    ]

    counts = await backfill(apply=True, batch_size=1)

    assert counts.stripped >= 3
    for session_id in sessions:
        assert await _contents(session_id) == [SKILLS_BLOCK + REST, "done"]
