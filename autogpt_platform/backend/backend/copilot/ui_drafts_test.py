import asyncio
import json
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import HTTPException
from pydantic import ValidationError

from backend.copilot.db import add_chat_message, create_chat_session
from backend.copilot.ui_drafts import UIDraftWrite, read_draft, write_draft
from backend.data import db


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
