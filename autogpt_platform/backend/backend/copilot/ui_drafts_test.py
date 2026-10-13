import asyncio
import json
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import HTTPException
from prisma.errors import PrismaError
from pydantic import ValidationError

from backend.copilot.db import add_chat_message, create_chat_session
from backend.copilot.ui_drafts import UIDraftWrite, read_draft, write_draft
from backend.copilot.ui_view_files import (
    UIDraftSaveRequest,
    UIFileStateWrite,
    read_file_state,
    save_draft_as_file,
    write_file_state,
)
from backend.data import db
from backend.data.workspace import OPENUI_MIME_TYPE, get_or_create_workspace
from backend.util.workspace import WorkspaceManager


@pytest_asyncio.fixture(loop_scope="session")
async def draft_user():
    owned_connection = not db.is_connected()
    if owned_connection:
        await db.connect()
    user_id = str(uuid4())
    await db.prisma.user.create(
        data={"id": user_id, "email": f"ui-draft-{user_id}@example.invalid"}
    )
    try:
        yield user_id
    finally:
        await db.prisma.user.delete(where={"id": user_id})
        if owned_connection:
            await db.disconnect()


async def _view(user_id: str) -> str:
    session = await create_chat_session(str(uuid4()), user_id)
    await add_chat_message(
        session.session_id,
        "tool",
        1,
        tool_call_id="view-one",
        content=json.dumps(
            {
                "type": "ui_rendered",
                "view_id": "trip",
                "source": 'root = Workspace("Trip", "", [], "trip")',
            }
        ),
    )
    return session.session_id


@pytest.mark.asyncio(loop_scope="session")
async def test_drafts_are_owner_scoped_and_survive_reload(draft_user):
    session = await _view(draft_user)
    try:
        first = await read_draft(draft_user, session, "trip", "view-one")
        assert first.revision == 0 and first.state == {}
        state = {
            "trip": {"notes": {"value": "Keep the bridge", "componentType": "Field"}}
        }
        saved = await write_draft(
            draft_user,
            session,
            "trip",
            UIDraftWrite(tool_call_id="view-one", expected_revision=0, state=state),
        )
        assert saved.revision == 1
        assert (
            await read_draft(draft_user, session, "trip", "view-one")
        ).state == state
        for operation in [
            read_draft("another-user", session, "trip", "view-one"),
            write_draft(
                "another-user",
                session,
                "trip",
                UIDraftWrite(tool_call_id="view-one", expected_revision=1, state={}),
            ),
            read_draft(draft_user, session, "different-view", "view-one"),
            write_draft(
                draft_user,
                session,
                "trip",
                UIDraftWrite(tool_call_id="invented", expected_revision=1, state={}),
            ),
        ]:
            with pytest.raises(HTTPException) as error:
                await operation
            assert error.value.status_code == 404
    finally:
        await db.prisma.chatsession.delete(where={"id": session})


@pytest.mark.asyncio(loop_scope="session")
async def test_simultaneous_saves_have_one_winner(draft_user):
    session = await _view(draft_user)
    try:
        results = await asyncio.gather(
            *[
                write_draft(
                    draft_user,
                    session,
                    "trip",
                    UIDraftWrite(
                        tool_call_id="view-one",
                        expected_revision=0,
                        state={"choice": value},
                    ),
                )
                for value in ["A", "B"]
            ],
            return_exceptions=True,
        )
        assert (
            sum(
                isinstance(result, HTTPException) and result.status_code == 409
                for result in results
            )
            == 1
        )
        saved = await read_draft(draft_user, session, "trip", "view-one")
        assert saved.revision == 1 and saved.state["choice"] in ["A", "B"]
    finally:
        await db.prisma.chatsession.delete(where={"id": session})


@pytest.mark.asyncio(loop_scope="session")
async def test_an_old_view_cannot_overwrite_a_new_revision(draft_user):
    session = await _view(draft_user)
    try:
        await write_draft(
            draft_user,
            session,
            "trip",
            UIDraftWrite(
                tool_call_id="view-one", expected_revision=0, state={"choice": "A"}
            ),
        )
        await add_chat_message(
            session,
            "tool",
            2,
            tool_call_id="view-two",
            content=json.dumps(
                {
                    "type": "ui_rendered",
                    "view_id": "trip",
                    "source": 'root = Workspace("Updated trip", "", [], "trip")',
                }
            ),
        )
        await write_draft(
            draft_user,
            session,
            "trip",
            UIDraftWrite(
                tool_call_id="view-two", expected_revision=1, state={"choice": "B"}
            ),
        )
        with pytest.raises(HTTPException) as error:
            await write_draft(
                draft_user,
                session,
                "trip",
                UIDraftWrite(
                    tool_call_id="view-one", expected_revision=2, state={"choice": "A"}
                ),
            )
        assert error.value.status_code == 409
        assert not (await read_draft(draft_user, session, "trip", "view-one")).writable
        assert (await read_draft(draft_user, session, "trip", "view-two")).state[
            "choice"
        ] == "B"
        await db.prisma.chatsession.delete(where={"id": session})
        rows = await db.prisma.query_raw(
            'SELECT * FROM "ChatUIDraft" WHERE "sessionId" = $1', session
        )
        assert rows == []
    finally:
        await db.prisma.chatsession.delete_many(where={"id": session})


def test_draft_size_and_revision_bounds():
    with pytest.raises(ValidationError):
        UIDraftWrite(tool_call_id="one", expected_revision=-1, state={})
    with pytest.raises(ValidationError):
        UIDraftWrite(
            tool_call_id="one", expected_revision=0, state={"large": "é" * 130_000}
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_published_revision_supersedes_old_card_before_either_saves(draft_user):
    session = await _view(draft_user)
    try:
        await add_chat_message(
            session,
            "tool",
            2,
            tool_call_id="view-two",
            content=json.dumps(
                {
                    "type": "ui_rendered",
                    "view_id": "trip",
                    "source": 'root = Workspace("Updated", "", [], "trip")',
                }
            ),
        )
        assert not (await read_draft(draft_user, session, "trip", "view-one")).writable
        assert (await read_draft(draft_user, session, "trip", "view-two")).writable
        with pytest.raises(HTTPException) as error:
            await write_draft(
                draft_user,
                session,
                "trip",
                UIDraftWrite(tool_call_id="view-one", expected_revision=0, state={}),
            )
        assert error.value.status_code == 409
        saved = await write_draft(
            draft_user,
            session,
            "trip",
            UIDraftWrite(
                tool_call_id="view-two", expected_revision=0, state={"latest": True}
            ),
        )
        assert saved.revision == 1
        await add_chat_message(
            session,
            "tool",
            3,
            tool_call_id="broken",
            content="not json ui_rendered view_id",
        )
        await add_chat_message(
            session,
            "tool",
            4,
            tool_call_id="unrelated",
            content=json.dumps(
                {"type": "ui_rendered", "view_id": "another", "source": "other"}
            ),
        )
        assert (await read_draft(draft_user, session, "trip", "view-two")).writable
    finally:
        await db.prisma.chatsession.delete(where={"id": session})


def _save(name: str = "Trip") -> UIDraftSaveRequest:
    return UIDraftSaveRequest(filename=name)


def _file_write(revision: int, checksum: str, **state) -> UIFileStateWrite:
    return UIFileStateWrite(
        expected_revision=revision, expected_checksum=checksum, state=state
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_draft_belongs_to_exactly_one_chat_or_file(draft_user):
    session = await _view(draft_user)
    try:
        for owner in [{}, {"sessionId": session, "fileId": str(uuid4())}]:
            with pytest.raises(PrismaError):
                await db.prisma.execute_raw(
                    'INSERT INTO "ChatUIDraft" (id, "sessionId", "fileId", "viewId", '
                    "sequence, source, state) VALUES ($1, $2, $3, $4, 0, $5, $6::jsonb)",
                    str(uuid4()),
                    owner.get("sessionId"),
                    owner.get("fileId"),
                    "trip",
                    "source",
                    "{}",
                )
    finally:
        await db.prisma.chatsession.delete(where={"id": session})


@pytest.mark.asyncio(loop_scope="session")
async def test_saving_moves_the_edits_onto_a_file_that_outlives_the_chat(
    draft_user,
):
    session = await _view(draft_user)
    state = {"trip": {"notes": {"value": "Bridge", "componentType": "Field"}}}
    await write_draft(
        draft_user,
        session,
        "trip",
        UIDraftWrite(tool_call_id="view-one", expected_revision=0, state=state),
    )
    saved = await save_draft_as_file(draft_user, session, "trip", _save())
    assert saved.file_name == "Trip.openui"
    assert saved.file_path == f"/sessions/{session}/Trip.openui"
    assert saved.file_uri == f"workspace://{saved.file_id}#{OPENUI_MIME_TYPE}"
    assert saved.view_id == "trip"

    again = await save_draft_as_file(draft_user, session, "trip", _save("Other"))
    assert again.file_id == saved.file_id
    rows = await db.prisma.chatuidraft.find_many(where={"viewId": "trip"})
    mine = [row for row in rows if row.fileId == saved.file_id]
    assert len(mine) == 1 and mine[0].sessionId is None
    assert not [row for row in rows if row.sessionId == session]

    in_chat = await read_draft(draft_user, session, "trip", "view-one")
    assert in_chat.file_id == saved.file_id and not in_chat.writable
    with pytest.raises(HTTPException) as error:
        await write_draft(
            draft_user,
            session,
            "trip",
            UIDraftWrite(tool_call_id="view-one", expected_revision=0, state={}),
        )
    assert error.value.status_code == 409

    on_file = await read_file_state(draft_user, saved.file_id)
    assert (on_file.state, on_file.revision) == (state, 1)
    assert on_file.source == 'root = Workspace("Trip", "", [], "trip")'
    assert on_file.checksum == saved.checksum
    assert on_file.origin_session_id == session

    await db.prisma.chatsession.delete(where={"id": session})
    orphaned = await read_file_state(draft_user, saved.file_id)
    assert orphaned.state == state and orphaned.origin_session_id is None


@pytest.mark.asyncio(loop_scope="session")
async def test_file_edits_need_the_current_revision_and_source(draft_user):
    session = await _view(draft_user)
    try:
        saved = await save_draft_as_file(draft_user, session, "trip", _save())
        fresh = await read_file_state(draft_user, saved.file_id)
        assert (fresh.revision, fresh.state) == (0, {})

        first = await write_file_state(
            draft_user, saved.file_id, _file_write(0, saved.checksum, choice="A")
        )
        assert first.revision == 1
        for stale in [
            _file_write(0, saved.checksum, choice="B"),
            _file_write(1, "not-the-checksum", choice="B"),
        ]:
            with pytest.raises(HTTPException) as error:
                await write_file_state(draft_user, saved.file_id, stale)
            assert error.value.status_code == 409

        workspace = await get_or_create_workspace(draft_user)
        manager = WorkspaceManager(draft_user, workspace.id, session)
        revised = 'root = Workspace("Trip, revised", "", [], "trip")'
        file = await manager.replace_file_content(
            saved.file_id,
            revised.encode(),
            mime_type=OPENUI_MIME_TYPE,
            metadata={"openui": {"viewId": "trip", "originSessionId": session}},
        )
        assert file.id == saved.file_id and file.checksum != saved.checksum
        with pytest.raises(HTTPException) as error:
            await write_file_state(
                draft_user, saved.file_id, _file_write(1, saved.checksum, choice="B")
            )
        assert error.value.status_code == 409

        current = await read_file_state(draft_user, saved.file_id)
        assert current.source == revised and current.state == {"choice": "A"}
        assert file.checksum is not None
        second = await write_file_state(
            draft_user, saved.file_id, _file_write(1, file.checksum, choice="B")
        )
        assert (second.revision, second.source) == (2, revised)
    finally:
        await db.prisma.chatsession.delete_many(where={"id": session})


@pytest.mark.asyncio(loop_scope="session")
async def test_saved_views_are_owner_scoped(draft_user):
    session = await _view(draft_user)
    try:
        saved = await save_draft_as_file(draft_user, session, "trip", _save())
        for operation in [
            read_file_state("another-user", saved.file_id),
            write_file_state(
                "another-user", saved.file_id, _file_write(0, saved.checksum)
            ),
            save_draft_as_file("another-user", session, "trip", _save()),
            save_draft_as_file(draft_user, session, "unknown-view", _save()),
        ]:
            with pytest.raises(HTTPException) as error:
                await operation
            assert error.value.status_code == 404
    finally:
        await db.prisma.chatsession.delete(where={"id": session})


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_saves_agree_on_one_file(draft_user):
    session = await _view(draft_user)
    try:
        results = await asyncio.gather(
            save_draft_as_file(draft_user, session, "trip", _save("One")),
            save_draft_as_file(draft_user, session, "trip", _save("Two")),
        )
        assert results[0].file_id == results[1].file_id
        rows = await db.prisma.chatuidraft.find_many(
            where={"fileId": results[0].file_id}
        )
        assert len(rows) == 1
    finally:
        await db.prisma.chatsession.delete(where={"id": session})


def test_saved_view_names_are_plain_openui_names():
    assert _save("Lisbon plan").filename == "Lisbon plan.openui"
    assert _save("plan.OPENUI").filename == "plan.OPENUI"
    for bad in ["", "  ", "../x", "a/b", ".."]:
        with pytest.raises(ValidationError):
            _save(bad)
