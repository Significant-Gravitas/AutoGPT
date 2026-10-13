"""``blank_dream_narratives`` against a real database: a hard forget blanks
the narrative a dream opens its session with (``dream/forget_copies.py``),
for the user's passes it names and nothing else: not the user's chat with
the narrative, another pass's, another user's or an ordinary chat."""

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest
from prisma.models import ChatMessage, ChatSession, User

from backend.util.json import SafeJson

from .db import blank_dream_narratives

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
