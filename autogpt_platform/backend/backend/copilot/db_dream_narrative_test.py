"""The narrative a dream opens its session with, against a real database.
``blank_dream_narratives``: a hard forget blanks it (``dream/forget_copies.py``)
for the user's passes it names, whose pass read a uuid it reached (the
session's copy of the read set, which outlives the pass's row), or, in the
forget's scope only, whose session has no record of what its pass read (no
read set, or one holding ``UNKNOWN_READS``), and nothing else: not the
user's chat with the narrative, another pass's, another scope's, another
user's or an ordinary chat. ``add_dream_narrative``: written only into the pass's
own session and only while its row is there, blank once the pass is marked,
in one statement holding the pass's row, so a mark landing on either side of
the write leaves no narrative behind (the review's cancelled-apply race)."""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone

import pytest
from prisma.enums import DreamPassRoute, DreamPassTrigger
from prisma.models import ChatMessage, ChatSession, DreamPass, Expert, User

from backend.data import db
from backend.data.dream_pass_models import UNKNOWN_READS
from backend.util.json import SafeJson

from . import db as chat
from .db import (
    DREAM_READ_UUIDS,
    add_dream_narrative,
    blank_dream_narratives,
    clear_session_pending_question,
    get_chat_session_metadata,
    set_session_pending_question,
)

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


async def _session(
    user_id: str, metadata: dict[str, object], expert_id: str | None = None
) -> str:
    """A session of the user's scope of *expert_id* (the account's when
    ``None``) that opens with ``_NARRATIVE``, then the user's reply and the
    assistant's answer, which repeats it."""
    session = await ChatSession.prisma().create(
        data={
            "userId": user_id,
            "metadata": SafeJson(metadata),
            **({"expertId": expert_id} if expert_id else {}),
        }
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
    read = {DREAM_READ_UUIDS: ["u9"]}
    named = await _session(owner, {"kind": "dream", "dream_pass_id": "p1", **read})
    unnamed = await _session(owner, {"kind": "dream", "dream_pass_id": "p2", **read})
    ordinary = await _session(owner, {"kind": "normal", "dream_pass_id": "p1"})
    theirs = await _session(stranger, {"kind": "dream", "dream_pass_id": "p1"})

    blanked = await blank_dream_narratives(
        owner, None, ["p1", "p-gone"], [], "[forgotten]"
    )

    assert blanked == 1
    assert await _contents(named) == ["[forgotten]", "Does she?", _NARRATIVE]
    for session_id in (unnamed, ordinary, theirs):
        assert await _contents(session_id) == [_NARRATIVE, "Does she?", _NARRATIVE]


async def test_no_passes_and_nothing_reached_writes_nothing(make_user):
    owner = await make_user()
    legacy = await _session(owner, {"kind": "dream", "dream_pass_id": "p1"})

    assert await blank_dream_narratives(owner, None, [], [], "[forgotten]") == 0
    assert await _contents(legacy) == [_NARRATIVE, "Does she?", _NARRATIVE]


async def test_a_narrative_whose_pass_read_a_reached_uuid_is_blanked_once(make_user):
    """Found by the session's copy of the read set, its pass's row gone or
    not; a session whose pass read none of it, or another user's, is left.
    A narrative already blank is not written again."""
    owner, stranger = await make_user(), await make_user()
    read = await _session(
        owner, {"kind": "dream", "dream_pass_id": "p1", DREAM_READ_UUIDS: ["u1", "u2"]}
    )
    unread = await _session(
        owner, {"kind": "dream", "dream_pass_id": "p2", DREAM_READ_UUIDS: ["u3"]}
    )
    theirs = await _session(
        stranger, {"kind": "dream", "dream_pass_id": "p3", DREAM_READ_UUIDS: ["u2"]}
    )

    assert await blank_dream_narratives(owner, None, [], ["u2"], "[forgotten]") == 1
    assert await blank_dream_narratives(owner, None, [], ["u2"], "[forgotten]") == 0

    assert await _contents(read) == ["[forgotten]", "Does she?", _NARRATIVE]
    for session_id in (unread, theirs):
        assert await _contents(session_id) == [_NARRATIVE, "Does she?", _NARRATIVE]


async def _expert(owner: str) -> str:
    expert = await Expert.prisma().create(
        data={
            "ownerUserId": owner,
            "name": "Scout",
            "role": "Research",
            "identity": "Finds things out.",
        }
    )
    return expert.id


@pytest.mark.parametrize("forget_in", ["account", "expert"])
async def test_a_narrative_of_unknown_provenance_goes_with_any_forget_in_its_scope(
    make_user, forget_in: str
):
    """A session written before sessions kept their pass's read set, or
    whose read set holds ``UNKNOWN_READS`` (its pass's, from before read
    sets were recorded), may rest on anything its pass read: any hard forget
    in its scope blanks it, whatever it reached. One whose read set is known
    and holds none of it is left, and so is every session of another scope,
    of another user, or that is not a dream."""
    owner, stranger = await make_user(), await make_user()
    expert = await _expert(owner)
    scope, other = (None, expert) if forget_in == "account" else (expert, None)

    def dream(read: object = None) -> dict[str, object]:
        known = {} if read is None else {DREAM_READ_UUIDS: read}
        return {"kind": "dream", "dream_pass_id": str(uuid.uuid4()), **known}

    no_read_set = await _session(owner, dream(), scope)
    marked = await _session(owner, dream([UNKNOWN_READS, "u1"]), scope)
    unreadable = await _session(owner, dream("u1"), scope)
    known = await _session(owner, dream(["u3"]), scope)
    other_scopes = await _session(owner, dream(), other)
    other_scopes_marked = await _session(owner, dream([UNKNOWN_READS]), other)
    theirs = await _session(stranger, dream())
    ordinary = await _session(owner, {"kind": "normal"}, scope)

    blanked = await blank_dream_narratives(owner, scope, [], ["u2"], "[forgotten]")

    assert blanked == 3
    for session_id in (no_read_set, marked, unreadable):
        assert await _contents(session_id) == ["[forgotten]", "Does she?", _NARRATIVE]
    for session_id in (known, other_scopes, other_scopes_marked, theirs, ordinary):
        assert await _contents(session_id) == [_NARRATIVE, "Does she?", _NARRATIVE]
    assert await blank_dream_narratives(owner, scope, [], ["u2"], "[forgotten]") == 0


async def test_a_pass_from_before_read_sets_marks_its_narrative_as_unknown(make_user):
    """A pass whose writer did not state its read set (one in flight across
    the deploy) reads ``UNKNOWN_READS``; its narrative's session takes it,
    and an unrelated forget in its scope blanks the narrative once the
    pass's row is gone."""
    owner = await make_user()
    pass_id = str(uuid.uuid4())
    await DreamPass.prisma().create(
        data={
            "id": pass_id,
            "userId": owner,
            "scopeKey": owner,
            "route": DreamPassRoute.SYNC,
            "trigger": DreamPassTrigger.CRON,
        }
    )
    session = await ChatSession.prisma().create(
        data={
            "userId": owner,
            "metadata": SafeJson({"kind": "dream", "dream_pass_id": pass_id}),
        }
    )
    assert await add_dream_narrative(session.id, pass_id, _NARRATIVE, "[forgotten]")
    stored = await ChatSession.prisma().find_unique(where={"id": session.id})
    assert stored is not None
    assert dict(stored.metadata)[DREAM_READ_UUIDS] == [UNKNOWN_READS]
    await DreamPass.prisma().delete(where={"id": pass_id})

    blanked = await blank_dream_narratives(
        owner, None, [], ["unrelated"], "[forgotten]"
    )

    assert blanked == 1
    assert await _contents(session.id) == ["[forgotten]"]


async def _dream_session(
    user_id: str, *, erased: bool, reads: tuple[str, ...] = ("fact-1", "episode-1")
) -> tuple[str, str]:
    """A pass of the user's that read *reads*, marked by a hard forget when
    ``erased``, and its empty dream session: the pass id and the session
    id."""
    pass_id = str(uuid.uuid4())
    await DreamPass.prisma().create(
        data={
            "id": pass_id,
            "userId": user_id,
            "scopeKey": user_id,
            "route": DreamPassRoute.SYNC,
            "trigger": DreamPassTrigger.CRON,
            "readUuids": list(reads),
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

    assert await add_dream_narrative(session_id, pass_id, _NARRATIVE, "[forgotten]")

    [message] = await ChatMessage.prisma().find_many(where={"sessionId": session_id})
    assert (message.role, message.sequence) == ("assistant", 0)
    assert message.content == ("[forgotten]" if erased else _NARRATIVE)
    session = await ChatSession.prisma().find_unique(where={"id": session_id})
    assert session is not None
    assert dict(session.metadata)[DREAM_READ_UUIDS] == ["fact-1", "episode-1"]


async def test_a_narrative_with_no_pass_row_is_not_written(make_user, caplog):
    """A pass whose row is not there (its insert failed, so it should never
    have run) has nothing a forget could find it by."""
    owner = await make_user()
    session = await ChatSession.prisma().create(
        data={
            "userId": owner,
            "metadata": SafeJson({"kind": "dream", "dream_pass_id": "p-no-row"}),
        }
    )

    with caplog.at_level(logging.WARNING, logger=chat.logger.name):
        written = await add_dream_narrative(
            session.id, "p-no-row", _NARRATIVE, "[forgotten]"
        )

    assert written is False
    assert await _contents(session.id) == []
    assert "its narrative is not written" in caplog.text


@pytest.mark.parametrize(
    "owner_is, metadata",
    [
        ("stranger", {"kind": "dream"}),
        ("owner", {"kind": "normal"}),
        ("owner", {"kind": "dream", "dream_pass_id": "another-pass"}),
    ],
    ids=["another_users", "not_a_dream", "another_passes"],
)
async def test_a_narrative_goes_only_into_its_passes_own_session(
    make_user, owner_is: str, metadata: dict[str, str]
):
    owner, stranger = await make_user(), await make_user()
    pass_id, _ = await _dream_session(owner, erased=False)
    session = await ChatSession.prisma().create(
        data={
            "userId": stranger if owner_is == "stranger" else owner,
            "metadata": SafeJson({"dream_pass_id": pass_id, **metadata}),
        }
    )

    assert not await add_dream_narrative(session.id, pass_id, _NARRATIVE, "[forgotten]")

    assert await _contents(session.id) == []
    stored = await ChatSession.prisma().find_unique(where={"id": session.id})
    assert stored is not None and DREAM_READ_UUIDS not in dict(stored.metadata)


async def test_the_read_set_outlives_the_sessions_reads_and_merges(make_user):
    """``ChatSessionMetadata`` leaves the key out (only the forget's SQL
    reads it), and every write to a session's metadata after its creation
    merges keys, so the copy stays for the forget."""
    owner = await make_user()
    pass_id, session_id = await _dream_session(owner, erased=False)
    assert await add_dream_narrative(session_id, pass_id, _NARRATIVE, "[forgotten]")

    await set_session_pending_question(
        session_id, owner, "Does she?", datetime.now(timezone.utc)
    )
    await clear_session_pending_question(session_id, owner)
    loaded = await get_chat_session_metadata(session_id)

    assert loaded is not None and loaded.metadata.kind == "dream"
    blanked = await blank_dream_narratives(owner, None, [], ["fact-1"], "[forgotten]")
    assert blanked == 1
    assert await _contents(session_id) == ["[forgotten]"]


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
    await blank_dream_narratives(owner, None, [pass_id], [], "[forgotten]")

    assert await _contents(session_id) == ["[forgotten]"]


async def test_the_blank_over_3000_dream_sessions_meets_database_deadline(make_user):
    """The database statement timeout bounds 3,000 dream sessions, each
    holding a 550-uuid read set, on both selective and all-match reads."""
    owner = await make_user()
    shared = "fact-read-by-every-pass"
    await db.execute_raw_with_schema(
        'INSERT INTO {schema_prefix}"ChatSession" ("id", "userId", "updatedAt", '
        '"metadata", "credentials", "successfulAgentRuns", '
        '"successfulAgentSchedules") '
        "SELECT gen_random_uuid()::text, $1, now(), jsonb_build_object("
        "'kind', 'dream', 'dream_pass_id', gen_random_uuid()::text, $2::text, "
        "to_jsonb(ARRAY(SELECT gen_random_uuid()::text "
        "FROM generate_series(1, 549) WHERE session.n > 0) || $3::text)), "
        "'{{}}'::jsonb, '{{}}'::jsonb, '{{}}'::jsonb "
        "FROM generate_series(1, 3000) AS session(n)",
        owner,
        DREAM_READ_UUIDS,
        shared,
    )
    await db.execute_raw_with_schema(
        'INSERT INTO {schema_prefix}"ChatMessage" ("id", "sessionId", "role", '
        '"sequence", "content") SELECT gen_random_uuid()::text, "id", '
        "'assistant', 0, $2 FROM {schema_prefix}\"ChatSession\" "
        'WHERE "userId" = $1',
        owner,
        _NARRATIVE,
    )
    unread = [str(uuid.uuid4()) for _ in range(1000)]

    assert await blank_dream_narratives(owner, None, [], unread, "[forgotten]") == 0
    blanked = await blank_dream_narratives(
        owner, None, [], [*unread[1:], shared], "[forgotten]"
    )

    assert blanked == 3000
