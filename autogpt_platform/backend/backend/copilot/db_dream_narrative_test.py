"""The narrative a dream opens its session with, against a real database.
``blank_dream_narratives``: a hard forget blanks it (``dream/forget_copies.py``)
for the user's passes it names and nothing else: not the user's chat with
the narrative, another pass's, another user's or an ordinary chat.
``add_dream_narrative``: written blank once the pass is marked, in one
statement holding the pass's row, so a mark landing on either side of the
write leaves no narrative behind (the review's cancelled-apply race)."""

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone

import pytest
from prisma.enums import DreamPassRoute, DreamPassTrigger
from prisma.models import ChatMessage, ChatSession, DreamPass, User

from backend.data import db
from backend.util.json import SafeJson

from .db import add_dream_narrative, blank_dream_narratives

pytestmark = pytest.mark.asyncio(loop_scope="session")

_NARRATIVE = "Last night I noticed Alice runs the Atlas launch."


@pytest.fixture
async def make_user() -> AsyncIterator[Callable[[], Awaitable[str]]]:
    created: list[str] = []

    async def make() -> str:
        user_id = str(uuid.uuid4())
        await User.prisma().create(
            data={
                "id": user_id,
                "email": f"dream-narrative-{user_id}@example.com",
                "topUpConfig": SafeJson({}),
                "timezone": "UTC",
            }
        )
        created.append(user_id)
        return user_id

    yield make
    await User.prisma().delete_many(where={"id": {"in": created}})


async def _session(user_id: str, metadata: dict[str, str]) -> str:
    """A session that opens with ``_NARRATIVE``, then the user's reply and the
    assistant's answer, which repeats it."""
    session = await ChatSession.prisma().create(
        data={"userId": user_id, "metadata": SafeJson(metadata)}
    )
    for sequence, (role, content) in enumerate(
        [
            ("assistant", _NARRATIVE),
            ("user", "Does she?"),
            ("assistant", _NARRATIVE),
        ]
    ):
        await ChatMessage.prisma().create(
            data={
                "sessionId": session.id,
                "role": role,
                "content": content,
                "sequence": sequence,
            }
        )
    return session.id


async def _contents(session_id: str) -> list[str | None]:
    messages = await ChatMessage.prisma().find_many(
        where={"sessionId": session_id}, order={"sequence": "asc"}
    )
    return [message.content for message in messages]


async def test_only_the_named_passes_narratives_are_blanked(make_user):
    owner, stranger = await make_user(), await make_user()
    named = await _session(owner, {"kind": "dream", "dream_pass_id": "p1"})
    unnamed = await _session(owner, {"kind": "dream", "dream_pass_id": "p2"})
    ordinary = await _session(owner, {"kind": "normal", "dream_pass_id": "p1"})
    theirs = await _session(stranger, {"kind": "dream", "dream_pass_id": "p1"})

    blanked = await blank_dream_narratives(owner, ["p1", "p-gone"], "[forgotten]")

    assert blanked == 1
    assert await _contents(named) == ["[forgotten]", "Does she?", _NARRATIVE]
    for session_id in (unnamed, ordinary, theirs):
        assert await _contents(session_id) == [_NARRATIVE, "Does she?", _NARRATIVE]


async def test_no_passes_writes_nothing(make_user):
    owner = await make_user()
    await _session(owner, {"kind": "dream", "dream_pass_id": "p1"})

    assert await blank_dream_narratives(owner, [], "[forgotten]") == 0


async def _dream_session(user_id: str, *, erased: bool) -> tuple[str, str]:
    """A pass of the user's, marked by a hard forget when ``erased``, and its
    empty dream session: the pass id and the session id."""
    pass_id = str(uuid.uuid4())
    await DreamPass.prisma().create(
        data={
            "id": pass_id,
            "userId": user_id,
            "scopeKey": user_id,
            "route": DreamPassRoute.SYNC,
            "trigger": DreamPassTrigger.CRON,
            **({"erasedAt": datetime_now()} if erased else {}),
        }
    )
    session = await ChatSession.prisma().create(
        data={
            "userId": user_id,
            "metadata": SafeJson({"kind": "dream", "dream_pass_id": pass_id}),
        }
    )
    return pass_id, session.id


def datetime_now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.mark.parametrize("erased", [False, True])
async def test_a_narrative_is_written_blank_once_its_pass_is_marked(
    make_user, erased: bool
):
    owner = await make_user()
    pass_id, session_id = await _dream_session(owner, erased=erased)

    await add_dream_narrative(session_id, pass_id, _NARRATIVE, "[forgotten]")

    [message] = await ChatMessage.prisma().find_many(where={"sessionId": session_id})
    assert (message.role, message.sequence) == ("assistant", 0)
    assert message.content == ("[forgotten]" if erased else _NARRATIVE)


async def test_a_narrative_with_no_pass_row_is_written_as_given(make_user):
    """A pass whose row was never inserted (the insert is best-effort)."""
    owner = await make_user()
    session = await ChatSession.prisma().create(
        data={"userId": owner, "metadata": SafeJson({"kind": "dream"})}
    )

    await add_dream_narrative(session.id, "p-no-row", _NARRATIVE, "[forgotten]")

    assert await _contents(session.id) == [_NARRATIVE]


async def test_a_mark_that_lands_first_holds_the_write_back_until_it_commits(
    make_user,
):
    """The forget marks the pass in a transaction; the narrative's write waits
    for it, then reads the mark and writes the placeholder."""
    owner = await make_user()
    pass_id, session_id = await _dream_session(owner, erased=False)

    async with db.transaction() as tx:
        await db.execute_raw_with_schema(
            'UPDATE {schema_prefix}"DreamPass" SET "erasedAt" = now() WHERE "id" = $1',
            pass_id,
            client=tx,
        )
        write = asyncio.create_task(
            add_dream_narrative(session_id, pass_id, _NARRATIVE, "[forgotten]")
        )
        await asyncio.sleep(0.5)
        assert not write.done(), "the write waits for the mark"
    await asyncio.wait_for(write, 10)

    assert await _contents(session_id) == ["[forgotten]"]


async def test_a_write_that_lands_first_is_found_by_the_forgets_blank(make_user):
    """The narrative's write holds the pass's row while it goes in: the mark
    waits for it, and the forget's blank of the narratives after the mark
    finds it."""
    owner = await make_user()
    pass_id, session_id = await _dream_session(owner, erased=False)

    async with db.transaction() as tx:
        await db.query_raw_with_schema(
            'SELECT 1 FROM {schema_prefix}"DreamPass" WHERE "id" = $1 FOR SHARE',
            pass_id,
            client=tx,
        )
        mark = asyncio.create_task(
            DreamPass.prisma().update(
                where={"id": pass_id}, data={"erasedAt": datetime_now()}
            )
        )
        await asyncio.sleep(0.5)
        assert not mark.done(), "the mark waits for the row the write holds"
        await tx.chatmessage.create(
            data={
                "sessionId": session_id,
                "role": "assistant",
                "sequence": 0,
                "content": _NARRATIVE,
            }
        )
    await asyncio.wait_for(mark, 10)
    await blank_dream_narratives(owner, [pass_id], "[forgotten]")

    assert await _contents(session_id) == ["[forgotten]"]
