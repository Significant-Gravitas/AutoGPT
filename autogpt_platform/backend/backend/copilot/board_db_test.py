"""Task board queries against the database: columns, which sessions are on
the board, the card facts, and the writes that must not bump ``updatedAt``."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, AsyncIterator

import pytest
import pytest_asyncio
from prisma.enums import ReviewStatus
from prisma.models import ChatMessage as PrismaChatMessage
from prisma.models import ChatSession as PrismaChatSession
from prisma.models import Expert, PendingHumanReview, User, UserWorkspaceFile

from backend.copilot import board_db
from backend.copilot.board import (
    BACKLOG,
    DEFAULT_COLUMNS,
    DONE,
    IN_PROGRESS,
    BoardColumn,
    new_column_key,
)
from backend.copilot.board_db import (
    get_board_columns,
    get_board_session_extras,
    list_board_sessions,
    list_session_files,
    mark_session_reassigned,
    save_board_columns,
    set_session_board,
    set_session_expert_if_unstarted,
)
from backend.copilot.board_reassign import reassign_brief_metadata
from backend.data import db
from backend.data.workspace import create_workspace_file, get_or_create_workspace
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer


@pytest_asyncio.fixture(loop_scope="session")
async def user_id(server: SpinTestServer) -> AsyncIterator[str]:
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"board-{user_id}@example.com"}
    )
    yield user_id
    await User.prisma().delete(where={"id": user_id})


@pytest.mark.asyncio(loop_scope="session")
async def test_a_user_without_a_board_gets_the_default_columns(user_id: str):
    columns = await get_board_columns(user_id)

    assert columns == DEFAULT_COLUMNS
    assert all(a is not b for a, b in zip(columns, DEFAULT_COLUMNS))


@pytest.mark.asyncio(loop_scope="session")
async def test_saved_columns_round_trip(user_id: str):
    custom = BoardColumn(key=new_column_key(), name="Waiting on Bob")
    columns = [*DEFAULT_COLUMNS[:2], custom, DEFAULT_COLUMNS[2]]

    saved = await save_board_columns(user_id, columns)

    assert await get_board_columns(user_id) == saved
    assert [c.key for c in saved] == [BACKLOG, IN_PROGRESS, custom.key, DONE]
    assert saved[2].name == "Waiting on Bob" and not saved[2].builtin


@pytest.mark.asyncio(loop_scope="session")
async def test_removing_a_column_moves_its_cards_to_backlog(user_id: str):
    custom = BoardColumn(key=new_column_key(), name="Review")
    await save_board_columns(user_id, [*DEFAULT_COLUMNS, custom])
    in_custom = await _session(user_id, board_column=custom.key)
    in_done = await _session(user_id, board_column=DONE)

    await save_board_columns(user_id, list(DEFAULT_COLUMNS))

    assert (await _row(in_custom)).boardColumn == BACKLOG
    assert (await _row(in_done)).boardColumn == DONE


@pytest.mark.asyncio(loop_scope="session")
async def test_a_placed_session_is_listed_however_old(user_id: str):
    placed = await _session(user_id, board_column=BACKLOG, board_position=0.0)
    await _age(placed, days=30)

    rows = await list_board_sessions(user_id, organization_id=None)

    listed = {r.session.session_id: r for r in rows}
    assert listed[placed].board_column == BACKLOG
    assert listed[placed].board_position == 0.0


@pytest.mark.asyncio(loop_scope="session")
async def test_the_row_cap_drops_done_cards_before_active_ones(
    user_id: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(board_db, "BOARD_SESSION_LIMIT", 2)
    await _session(user_id, board_column=DONE, board_position=0.0)
    await _session(user_id, board_column=DONE, board_position=1.0)
    active = await _session(user_id, board_column=IN_PROGRESS, board_position=0.0)

    rows = await list_board_sessions(user_id, organization_id=None)

    assert len(rows) == 2
    assert active in [r.session.session_id for r in rows]


@pytest.mark.asyncio(loop_scope="session")
async def test_an_unplaced_session_is_listed_only_while_recent(user_id: str):
    recent = await _session(user_id)
    stale = await _session(user_id)
    await _age(stale, days=8)

    default = await list_board_sessions(user_id, organization_id=None)
    with_older = await list_board_sessions(
        user_id, organization_id=None, include_older=True
    )

    default_ids = [r.session.session_id for r in default]
    older_ids = [r.session.session_id for r in with_older]
    assert recent in default_ids
    assert stale not in default_ids
    assert stale in older_ids
    assert (
        next(r for r in with_older if r.session.session_id == stale).board_column
        is None
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_dream_and_builder_sessions_are_not_on_the_board(user_id: str):
    dream = await _session(user_id, metadata={"kind": "dream"})
    builder = await _session(user_id, metadata={"builder_graph_id": "graph-1"})

    rows = await list_board_sessions(user_id, organization_id=None, include_older=True)

    listed = [r.session.session_id for r in rows]
    assert dream not in listed
    assert builder not in listed


@pytest.mark.asyncio(loop_scope="session")
async def test_a_hidden_kickoff_does_not_start_a_session(user_id: str):
    kickoff_only = await _session(user_id)
    await _message(
        kickoff_only, "user", 0, metadata={"hidden": True, "kind": "expert_kickoff"}
    )
    written_in = await _session(user_id)
    await _message(written_in, "user", 0)

    extras = await get_board_session_extras(user_id, [kickoff_only, written_in])

    assert extras[kickoff_only].started is False
    assert extras[written_in].started is True


@pytest.mark.asyncio(loop_scope="session")
async def test_a_reassign_brief_starts_a_session(user_id: str):
    reassigned = await _session(user_id)
    await _message(reassigned, "user", 0, metadata=reassign_brief_metadata("e1"))

    extras = await get_board_session_extras(user_id, [reassigned])

    assert extras[reassigned].started is True


@pytest.mark.asyncio(loop_scope="session")
async def test_a_pending_question_needs_you(user_id: str):
    asking = await _session(
        user_id,
        metadata={"pending_question": {"text": "Which plan?", "asked_at": "x"}},
    )
    quiet = await _session(user_id, metadata={"pending_question": None})

    extras = await get_board_session_extras(user_id, [asking, quiet])

    assert extras[asking].needs_you is True
    assert extras[quiet].needs_you is False


@pytest.mark.asyncio(loop_scope="session")
async def test_a_waiting_review_needs_you(user_id: str):
    reviewing = await _session(user_id)
    await PendingHumanReview.prisma().create(
        data={
            "nodeExecId": f"copilot-board-{uuid.uuid4()}",
            "userId": user_id,
            "chatSessionId": reviewing,
            "payload": SafeJson({"foo": "bar"}),
            "status": ReviewStatus.WAITING,
        }
    )

    extras = await get_board_session_extras(user_id, [reviewing])

    assert extras[reviewing].needs_you is True


@pytest.mark.asyncio(loop_scope="session")
async def test_the_last_line_is_the_first_sentence_of_the_latest_reply(user_id: str):
    session = await _session(user_id)
    await _message(session, "assistant", 0, content="Earlier reply.")
    await _message(session, "user", 1)
    await _message(session, "assistant", 2, content="Done. More.")
    await _message(session, "assistant", 3, content="")

    extras = await get_board_session_extras(user_id, [session])

    assert extras[session].last_line == "Done."


@pytest.mark.asyncio(loop_scope="session")
async def test_the_file_count_skips_deleted_files(user_id: str):
    session = await _session(user_id)
    workspace = await get_or_create_workspace(user_id)
    for name in ("live.txt", "gone.txt"):
        stored = await create_workspace_file(
            workspace_id=workspace.id,
            file_id=str(uuid.uuid4()),
            name=name,
            path=f"/sessions/{session}/{name}",
            storage_path=f"{workspace.id}/{name}",
            mime_type="text/plain",
            size_bytes=1,
        )
        if name == "gone.txt":
            await UserWorkspaceFile.prisma().update(
                where={"id": stored.id},
                data={"isDeleted": True, "deletedAt": datetime.now(UTC)},
            )

    extras = await get_board_session_extras(user_id, [session])

    assert extras[session].file_count == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_session_files_list_live_files_under_that_session_only(user_id: str):
    session = await _session(user_id)
    other = await _session(user_id)
    workspace = await get_or_create_workspace(user_id)
    stored = {}
    for owner, name in ((session, "live.txt"), (session, "gone.txt"), (other, "x")):
        stored[name] = await create_workspace_file(
            workspace_id=workspace.id,
            file_id=str(uuid.uuid4()),
            name=name,
            path=f"/sessions/{owner}/{name}",
            storage_path=f"{workspace.id}/{owner}/{name}",
            mime_type="text/plain",
            size_bytes=1,
        )
    await UserWorkspaceFile.prisma().update(
        where={"id": stored["gone.txt"].id},
        data={"isDeleted": True, "deletedAt": datetime.now(UTC)},
    )

    assert await list_session_files(user_id, session) == [
        ("live.txt", stored["live.txt"].id)
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_extras_leave_out_another_users_session(user_id: str):
    stranger = await _stranger()
    try:
        theirs = await _session(stranger)

        assert await get_board_session_extras(user_id, [theirs]) == {}
    finally:
        await User.prisma().delete(where={"id": stranger})


@pytest.mark.asyncio(loop_scope="session")
async def test_the_assignee_changes_while_only_a_hidden_kickoff_exists(user_id: str):
    expert = await _expert(user_id)
    session = await _session(user_id)
    await _message(session, "user", 0, metadata={"hidden": True})

    outcome = await set_session_expert_if_unstarted(
        session, user_id, expert, organization_id=None, team_id=None
    )

    assert outcome == "ok"
    assert (await _row(session)).expertId == expert


@pytest.mark.asyncio(loop_scope="session")
async def test_the_assignee_is_fixed_once_the_user_has_written(user_id: str):
    expert = await _expert(user_id)
    session = await _session(user_id)
    await _message(session, "user", 0)

    outcome = await set_session_expert_if_unstarted(
        session, user_id, expert, organization_id=None, team_id=None
    )

    assert outcome == "started"
    assert (await _row(session)).expertId is None


@pytest.mark.asyncio(loop_scope="session")
async def test_the_assignee_of_another_users_session_cannot_change(user_id: str):
    stranger = await _stranger()
    try:
        theirs = await _session(stranger)

        outcome = await set_session_expert_if_unstarted(
            theirs, user_id, None, organization_id=None, team_id=None
        )

        assert outcome == "not_found"
    finally:
        await User.prisma().delete(where={"id": stranger})


@pytest.mark.asyncio(loop_scope="session")
async def test_moving_a_card_does_not_touch_updated_at(user_id: str):
    session = await _session(user_id)
    await _age(session, days=2)
    before = (await _row(session)).updatedAt

    moved = await set_session_board(
        session, user_id, column=IN_PROGRESS, position=-512.0
    )

    after = await _row(session)
    assert moved is True
    assert after.boardColumn == IN_PROGRESS
    assert after.boardPosition == -512.0
    assert after.updatedAt == before


@pytest.mark.asyncio(loop_scope="session")
async def test_a_reassigned_card_moves_to_the_top_of_done_and_keeps_its_question(
    user_id: str,
):
    await _session(user_id, board_column=DONE, board_position=5.0)
    question = {"text": "Which plan?", "asked_at": "2026-10-12T00:00:00+00:00"}
    old = await _session(
        user_id,
        board_column=IN_PROGRESS,
        board_position=0.0,
        metadata={"pending_question": question},
    )

    await mark_session_reassigned(
        old, user_id, to_session_id="new-session", to_expert_id="expert-2"
    )

    row = await _row(old)
    assert row.boardColumn == DONE
    assert row.boardPosition == 5.0 - 1024
    assert isinstance(row.metadata, dict)
    assert row.metadata["pending_question"] == question
    assert row.metadata["reassigned_to_session_id"] == "new-session"
    assert row.metadata["reassigned_to_expert_id"] == "expert-2"


async def _session(
    user_id: str,
    *,
    metadata: dict[str, Any] | None = None,
    board_column: str | None = None,
    board_position: float | None = None,
) -> str:
    row = await PrismaChatSession.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "userId": user_id,
            "credentials": SafeJson({}),
            "successfulAgentRuns": SafeJson({}),
            "successfulAgentSchedules": SafeJson({}),
            "metadata": SafeJson(metadata or {}),
            "boardColumn": board_column,
            "boardPosition": board_position,
        }
    )
    return row.id


async def _message(
    session_id: str,
    role: str,
    sequence: int,
    *,
    content: str = "hello",
    metadata: dict[str, Any] | None = None,
) -> None:
    await PrismaChatMessage.prisma().create(
        data={
            "sessionId": session_id,
            "role": role,
            "content": content,
            "sequence": sequence,
            **({"metadata": SafeJson(metadata)} if metadata is not None else {}),
        }
    )


async def _age(session_id: str, *, days: int) -> None:
    await db.execute_raw_with_schema(
        'UPDATE {schema_prefix}"ChatSession" '
        "SET \"updatedAt\" = ($2::timestamptz AT TIME ZONE 'UTC') "
        'WHERE "id" = $1',
        session_id,
        datetime.now(UTC) - timedelta(days=days),
    )


async def _row(session_id: str) -> PrismaChatSession:
    row = await PrismaChatSession.prisma().find_unique(where={"id": session_id})
    assert row is not None
    return row


async def _expert(user_id: str) -> str:
    row = await Expert.prisma().create(
        data={
            "ownerUserId": user_id,
            "name": "Nova",
            "role": "",
            "identity": "nova",
            "skills": [],
        }
    )
    return row.id


async def _stranger() -> str:
    stranger = str(uuid.uuid4())
    await User.prisma().create(
        data={"id": stranger, "email": f"board-{stranger}@example.com"}
    )
    return stranger
