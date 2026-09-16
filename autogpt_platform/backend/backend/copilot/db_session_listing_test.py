"""Tests for copilot.db session LIST paths — dream-session exclusion.

The dream pass creates ChatSession rows with ``metadata.kind == "dream"``.
Those must be hidden from the user-facing LIST paths (chat sidebar,
pagination count, /search/global title search) while staying fetchable
by id. The whole risk is SQL NULL semantics: most sessions predate the
``kind`` metadata key, so ``metadata->>'kind'`` is NULL for them and a
naive ``<>`` comparison would silently hide every normal chat.

Mock-based tests pin the exact WHERE construction; the integration tests
at the bottom prove the NULL semantics against the real Postgres.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from prisma.enums import ResourceVisibility
from prisma.models import ChatSession as PrismaChatSession
from prisma.types import ChatSessionCreateInput

from backend.copilot.db import (
    _PENDING_QUESTION_SESSION_COLUMNS,
    clear_session_pending_question,
    create_chat_session,
    get_chat_session_metadata,
    get_sessions_with_pending_question,
    get_user_chat_sessions,
    get_user_session_count,
    set_session_pending_question,
    update_chat_session_title,
    user_has_any_session,
)
from backend.copilot.model import ChatSessionMetadata
from backend.util.json import SafeJson

_NULL_SAFE_DREAM_FILTER = "(metadata->>'kind' IS DISTINCT FROM 'dream')"
_RAW_QUERY_TARGET = "backend.copilot.db.db.query_raw_with_schema"


def _make_prisma_session(session_id: str = "sess-1") -> PrismaChatSession:
    now = datetime.now(UTC)
    return PrismaChatSession(
        id=session_id,
        createdAt=now,
        updatedAt=now,
        userId="user-1",
        title="hello",
        credentials="{}",
        successfulAgentRuns="{}",
        successfulAgentSchedules="{}",
        metadata="{}",
        totalPromptTokens=0,
        totalCompletionTokens=0,
        chatStatus="idle",
        isPinned=False,
        isShared=False,
        shareToken=None,
        sharedAt=None,
        autoShareExecutions=False,
        visibility=ResourceVisibility.PRIVATE,
    )


# ---------- WHERE construction (mocked raw query) ----------


@pytest.mark.asyncio
async def test_list_query_excludes_dreams_with_null_safe_operator():
    """The list WHERE clause uses IS DISTINCT FROM, never a bare inequality.

    ``NULL <> 'dream'`` evaluates to NULL for legacy rows without a
    ``kind`` key, which would hide every normal chat from the sidebar.
    """
    raw = AsyncMock(return_value=[])
    with patch(_RAW_QUERY_TARGET, raw):
        result = await get_user_chat_sessions("u1")

    assert result == []
    query = raw.call_args.args[0]
    assert _NULL_SAFE_DREAM_FILTER in query
    assert "<>" not in query
    assert "!=" not in query
    assert "ILIKE" not in query
    assert raw.call_args.args[1:] == ("u1", 50, 0)
    assert raw.call_args.kwargs["model"] is PrismaChatSession


@pytest.mark.asyncio
async def test_list_query_maps_raw_rows_to_chat_session_info():
    raw = AsyncMock(return_value=[_make_prisma_session("sess-42")])
    with patch(_RAW_QUERY_TARGET, raw):
        result = await get_user_chat_sessions("user-1")

    assert len(result) == 1
    assert result[0].session_id == "sess-42"
    assert result[0].user_id == "user-1"
    assert result[0].metadata.kind == "normal"


@pytest.mark.asyncio
async def test_title_search_escapes_like_wildcards_and_keeps_dream_filter():
    """User-supplied search text must match literally (no %/_ wildcards)."""
    raw = AsyncMock(return_value=[])
    with patch(_RAW_QUERY_TARGET, raw):
        await get_user_chat_sessions(
            "u1", limit=10, offset=5, title_contains="50%_done\\"
        )

    query = raw.call_args.args[0]
    assert _NULL_SAFE_DREAM_FILTER in query
    assert '"title" ILIKE $2' in query
    assert "LIMIT $3 OFFSET $4" in query
    assert raw.call_args.args[1:] == ("u1", "%50\\%\\_done\\\\%", 10, 5)


@pytest.mark.asyncio
async def test_autopilot_only_list_query_requires_null_expert_id():
    raw = AsyncMock(return_value=[])
    with patch(_RAW_QUERY_TARGET, raw):
        await get_user_chat_sessions("u1", autopilot_only=True)

    query = raw.call_args.args[0]
    assert '"expertId" IS NULL' in query
    assert raw.call_args.args[1:] == ("u1", 50, 0)


@pytest.mark.asyncio
async def test_autopilot_only_count_query_requires_null_expert_id():
    raw = AsyncMock(return_value=[{"count": 3}])
    with patch(_RAW_QUERY_TARGET, raw):
        count = await get_user_session_count("u1", autopilot_only=True)

    assert count == 3
    query = raw.call_args.args[0]
    assert '"expertId" IS NULL' in query
    assert raw.call_args.args[1:] == ("u1",)


@pytest.mark.asyncio
async def test_autopilot_only_and_expert_filter_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        await get_user_chat_sessions("u1", expert_id="expert-1", autopilot_only=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        await get_user_session_count("u1", expert_id="expert-1", autopilot_only=True)


@pytest.mark.asyncio
async def test_empty_expert_filter_is_rejected_consistently():
    with pytest.raises(ValueError, match="non-empty"):
        await get_user_chat_sessions("u1", expert_id="")
    with pytest.raises(ValueError, match="non-empty"):
        await get_user_session_count("u1", expert_id="")


@pytest.mark.asyncio
async def test_count_query_uses_same_dream_exclusion_as_list():
    """Sidebar pagination count must stay consistent with the visible list."""
    raw = AsyncMock(return_value=[{"count": 7}])
    with patch(_RAW_QUERY_TARGET, raw):
        count = await get_user_session_count("u1")

    assert count == 7
    query = raw.call_args.args[0]
    assert _NULL_SAFE_DREAM_FILTER in query
    assert raw.call_args.args[1:] == ("u1",)


@pytest.mark.asyncio
async def test_count_returns_zero_when_query_yields_no_rows():
    raw = AsyncMock(return_value=[])
    with patch(_RAW_QUERY_TARGET, raw):
        assert await get_user_session_count("u1") == 0


@pytest.mark.asyncio
async def test_presence_query_stops_at_the_first_row_and_keeps_the_dream_filter():
    """The greeting gate reads presence, so it must not scan every session.

    A regression in either half — the dream exclusion or the ``LIMIT 1`` —
    is silent at the call site: ``_retire_greeting_if_chatted`` swallows
    failures and answers "no", so every new user would keep a greeting
    they have already outgrown.
    """
    raw = AsyncMock(return_value=[{"?column?": 1}])
    with patch(_RAW_QUERY_TARGET, raw):
        assert await user_has_any_session("u1") is True

    query = raw.call_args.args[0]
    assert _NULL_SAFE_DREAM_FILTER in query
    assert "LIMIT 1" in query
    assert raw.call_args.args[1:] == ("u1",)


@pytest.mark.asyncio
async def test_presence_is_false_when_the_user_owns_no_visible_session():
    raw = AsyncMock(return_value=[])
    with patch(_RAW_QUERY_TARGET, raw):
        assert await user_has_any_session("u1") is False


# ---------- NULL semantics against the real database ----------


async def _create_legacy_session_without_kind_key(user_id: str) -> str:
    """Insert a pre-dream-PR row whose metadata JSON has no ``kind`` key."""
    legacy_id = str(uuid4())
    await PrismaChatSession.prisma().create(
        data=ChatSessionCreateInput(
            id=legacy_id,
            userId=user_id,
            credentials=SafeJson({}),
            successfulAgentRuns=SafeJson({}),
            successfulAgentSchedules=SafeJson({}),
            metadata=SafeJson({"dry_run": False}),
        )
    )
    return legacy_id


@pytest.mark.asyncio(loop_scope="session")
async def test_dream_sessions_hidden_from_list_but_fetchable_by_id(
    setup_test_user, test_user_id
):
    """Legacy rows (no ``kind`` key) stay listed; dream rows are hidden
    from the list yet still resolvable by session_id (Memory Visualizer
    and dream flows read them directly)."""
    normal = await create_chat_session(str(uuid4()), test_user_id)
    legacy_id = await _create_legacy_session_without_kind_key(test_user_id)
    dream = await create_chat_session(
        str(uuid4()),
        test_user_id,
        metadata=ChatSessionMetadata(kind="dream", dream_pass_id="pass-1"),
    )

    listed_ids = {
        s.session_id for s in await get_user_chat_sessions(test_user_id, limit=10_000)
    }
    assert normal.session_id in listed_ids
    assert legacy_id in listed_ids
    assert dream.session_id not in listed_ids

    fetched = await get_chat_session_metadata(dream.session_id)
    assert fetched is not None
    assert fetched.metadata.kind == "dream"
    assert fetched.metadata.dream_pass_id == "pass-1"


@pytest.mark.asyncio(loop_scope="session")
async def test_session_count_stays_consistent_with_visible_list(
    setup_test_user, test_user_id
):
    """Creating a dream session must not bump the pagination total."""
    count_before = await get_user_session_count(test_user_id)

    await create_chat_session(str(uuid4()), test_user_id)
    await create_chat_session(
        str(uuid4()),
        test_user_id,
        metadata=ChatSessionMetadata(kind="dream", dream_pass_id="pass-2"),
    )

    count_after = await get_user_session_count(test_user_id)
    assert count_after == count_before + 1

    listed = await get_user_chat_sessions(test_user_id, limit=10_000)
    assert count_after == len(listed)


@pytest.mark.asyncio(loop_scope="session")
async def test_title_search_matches_literally_and_skips_dream_sessions(
    setup_test_user, test_user_id
):
    """ILIKE search is case-insensitive, treats ``_`` literally, and never
    surfaces dream sessions even on an exact title match."""
    marker = f"Ndl{uuid4().hex[:8]}"

    normal = await create_chat_session(str(uuid4()), test_user_id)
    assert await update_chat_session_title(
        normal.session_id, test_user_id, f"{marker}_alpha plan"
    )
    decoy = await create_chat_session(str(uuid4()), test_user_id)
    assert await update_chat_session_title(
        decoy.session_id, test_user_id, f"{marker}Xalpha plan"
    )
    dream = await create_chat_session(
        str(uuid4()),
        test_user_id,
        metadata=ChatSessionMetadata(kind="dream", dream_pass_id="pass-3"),
    )
    assert await update_chat_session_title(
        dream.session_id, test_user_id, f"{marker}_alpha dream"
    )

    results = await get_user_chat_sessions(
        test_user_id, title_contains=f"{marker}_ALPHA"
    )
    result_ids = {s.session_id for s in results}
    assert normal.session_id in result_ids
    assert decoy.session_id not in result_ids
    assert dream.session_id not in result_ids


# ---------- get_sessions_with_pending_question (Home "Needs You") ----------


async def _ordered_pending_question_ids(user_id: str) -> list[str]:
    """Home's pending-question feed, newest first, as the query returned it.

    The limit is far above the module's own seeding so an ordering assertion
    can never be cut off by rows earlier tests left behind for this user.
    """
    sessions = await get_sessions_with_pending_question(user_id, limit=1000)
    return [s.session_id for s in sessions]


async def _pending_question_ids(user_id: str) -> set[str]:
    """Membership-only view of the same feed, for presence assertions."""
    return set(await _ordered_pending_question_ids(user_id))


@pytest.mark.asyncio
async def test_pending_question_query_avoids_select_star_and_excludes_delegated():
    """The Home pending-question query must select only the columns
    ``ChatSessionInfo.from_db`` reads (plus the columns the generated
    ``PrismaChatSession`` model requires with no default) — not ``SELECT *``
    — and must exclude delegated sub-session threads (but NOT handed-off
    ones, which own their task and whose question is the only path back to
    the user)."""
    raw = AsyncMock(return_value=[])
    with patch(_RAW_QUERY_TARGET, raw):
        result = await get_sessions_with_pending_question("u1", limit=5)

    assert result == []
    query = raw.call_args.args[0]
    assert "SELECT *" not in query
    assert '"id"' in query
    assert '"metadata"' in query
    assert "\"metadata\" ->> 'delegated_by_session_id' IS NULL" in query
    # A handoff sets ``delegated_by_session_id`` too, so the delegated test
    # alone would swallow every handed-off thread. The re-admitting arm has to
    # be in the predicate, not merely absent from it.
    assert "\"metadata\" ->> 'handed_off_from_expert_id' IS NOT NULL" in query
    # NULL-safe form: must use ``->>`` (text extraction), never a bare ``->``
    # comparison, which would treat every session's default explicit JSON
    # ``null`` as "present" and hide every normal session.
    assert "\"metadata\" -> 'delegated_by_session_id' IS NULL" not in query
    assert raw.call_args.args[1:] == ("u1", 5)
    assert raw.call_args.kwargs["model"] is PrismaChatSession


def test_pending_question_projection_columns_exist_in_schema():
    """Guard the hand-maintained projection in the Home pending-question query.

    ``_PENDING_QUESTION_SESSION_COLUMNS`` mirrors the ChatSession schema by
    hand, so a ``schema.prisma`` rename — or a new column with no default on
    the generated ``PrismaChatSession`` model, which the raw query hydrates —
    would otherwise only surface as a runtime query error on Home.
    """
    schema = (Path(__file__).parents[2] / "schema.prisma").read_text()
    model = re.search(r"^model ChatSession \{(.*?)^\}", schema, re.S | re.M)
    assert model is not None, "ChatSession model not found in schema.prisma"
    fields = set(re.findall(r"^\s{2}(\w+)", model.group(1), re.M))

    projected = set(re.findall(r'"(\w+)"', _PENDING_QUESTION_SESSION_COLUMNS))
    assert projected <= fields

    required = {
        name
        for name, field in PrismaChatSession.model_fields.items()
        if field.is_required()
    }
    assert required <= projected


@pytest.mark.asyncio(loop_scope="session")
async def test_pending_question_excludes_delegated_but_keeps_handed_off(
    setup_test_user, test_user_id
):
    """A *delegated* sub's question must not surface on Home — its caller is
    still waiting and owns reporting back. A *handed-off* thread must, since
    the handoff gave it the task outright and it is told to ask the user
    directly; hiding its question is how the request dies in silence.

    Also proves the exclusion predicate is NULL-safe: every session
    (including the normal one here) persists ``delegated_by_session_id`` as an
    explicit JSON ``null`` by default, and that must NOT be mistaken for
    "is a delegated sub".
    """
    now = datetime.now(UTC)

    normal = await create_chat_session(str(uuid4()), test_user_id)
    await set_session_pending_question(
        normal.session_id, test_user_id, "What's the budget?", now
    )

    delegated = await create_chat_session(
        str(uuid4()),
        test_user_id,
        metadata=ChatSessionMetadata(delegated_by_session_id=normal.session_id),
    )
    await set_session_pending_question(
        delegated.session_id, test_user_id, "Which API key?", now
    )

    # Mirror what ``handoff_to_expert._transfer`` actually writes: a handoff
    # records the delegation fields *as well as* the handoff marker. Building
    # this session with the marker alone made the assertion below vacuous —
    # it tested a shape the product never produces, and the real one was
    # excluded by the delegated-sub filter.
    handed_off = await create_chat_session(
        str(uuid4()),
        test_user_id,
        metadata=ChatSessionMetadata(
            delegated_by_session_id=normal.session_id,
            delegated_by_expert_id="expert-1",
            handed_off_from_expert_id="expert-1",
        ),
    )
    await set_session_pending_question(
        handed_off.session_id, test_user_id, "Confirm the vendor?", now
    )

    pending_ids = await _pending_question_ids(test_user_id)
    assert normal.session_id in pending_ids
    assert handed_off.session_id in pending_ids
    assert delegated.session_id not in pending_ids


@pytest.mark.asyncio(loop_scope="session")
async def test_pending_questions_come_back_newest_asked_first(
    setup_test_user, test_user_id
):
    """``ORDER BY ... asked_at DESC`` is a lexicographic *text* sort over the
    stored ISO strings; it only matches chronological order because every
    writer emits a fixed-width, zero-padded UTC ``isoformat()``. Seed three
    clearly separated timestamps — deliberately out of insertion order — and
    prove the newest question leads.
    """
    stamps = {
        "middle": datetime(2031, 5, 9, 13, 45, 6, tzinfo=UTC),
        "oldest": datetime(2031, 1, 2, 3, 4, 5, tzinfo=UTC),
        "newest": datetime(2031, 11, 30, 23, 59, 59, tzinfo=UTC),
    }
    seeded: dict[str, str] = {}
    for label, asked_at in stamps.items():
        session = await create_chat_session(str(uuid4()), test_user_id)
        await set_session_pending_question(
            session.session_id, test_user_id, f"{label}?", asked_at
        )
        seeded[label] = session.session_id

    ordered = await _ordered_pending_question_ids(test_user_id)
    mine = [sid for sid in ordered if sid in set(seeded.values())]
    assert mine == [seeded["newest"], seeded["middle"], seeded["oldest"]]


@pytest.mark.asyncio(loop_scope="session")
async def test_clearing_a_pending_question_removes_the_key(
    setup_test_user, test_user_id
):
    session = await create_chat_session(str(uuid4()), test_user_id)
    await set_session_pending_question(
        session.session_id, test_user_id, "Which vendor?", datetime.now(UTC)
    )
    assert session.session_id in await _pending_question_ids(test_user_id)

    await clear_session_pending_question(session.session_id, test_user_id)

    assert session.session_id not in await _pending_question_ids(test_user_id)
    fetched = await get_chat_session_metadata(session.session_id)
    assert fetched is not None
    assert fetched.metadata.pending_question is None


@pytest.mark.asyncio(loop_scope="session")
async def test_clearing_a_pending_question_is_scoped_to_the_owner(
    setup_test_user, test_user_id
):
    """The ``userId`` predicate is the whole authorization story for this
    write: without it any caller could silently dismiss another user's
    "Needs You" item. A non-owner must be a no-op, not a partial edit.
    """
    session = await create_chat_session(str(uuid4()), test_user_id)
    await set_session_pending_question(
        session.session_id, test_user_id, "Which vendor?", datetime.now(UTC)
    )

    await clear_session_pending_question(session.session_id, str(uuid4()))

    fetched = await get_chat_session_metadata(session.session_id)
    assert fetched is not None
    assert fetched.metadata.pending_question is not None
    assert fetched.metadata.pending_question.text == "Which vendor?"
    assert session.session_id in await _pending_question_ids(test_user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_pending_question_excludes_explicit_json_null(
    setup_test_user, test_user_id
):
    """Every session persists ``metadata.pending_question`` as an explicit
    JSON ``null`` by default (a JSON null present under the key, not a SQL
    NULL). The filter must use ``jsonb_typeof(...) = 'object'`` — an
    ``IS NOT NULL`` check on the raw ``->`` extraction would treat that
    default null as "present" and surface every ordinary session on Home.
    """
    no_question = await create_chat_session(str(uuid4()), test_user_id)

    with_question = await create_chat_session(str(uuid4()), test_user_id)
    await set_session_pending_question(
        with_question.session_id, test_user_id, "Which vendor?", datetime.now(UTC)
    )

    pending_ids = {
        s.session_id
        for s in await get_sessions_with_pending_question(test_user_id, limit=100)
    }
    assert no_question.session_id not in pending_ids
    assert with_question.session_id in pending_ids
