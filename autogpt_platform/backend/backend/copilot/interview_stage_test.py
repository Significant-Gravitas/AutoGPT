"""How an onboarding interview's stage is written: forward only, scoped to the
owner and the interview kind, and kept by the cache across a stale turn."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from backend.copilot import db as chat_db_module
from backend.copilot import model
from backend.copilot.model import (
    INTERVIEW_STAGE_CLOSED,
    INTERVIEW_STAGE_HANDED_OFF,
    INTERVIEW_STAGE_WRAP_UP_ISSUED,
    INTERVIEW_STAGES,
    ONBOARDING_INTERVIEW_KIND,
    ChatSession,
    ChatSessionInfo,
    ChatSessionMetadata,
)

_EXECUTE = "backend.copilot.db.db.execute_raw_with_schema"


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", INTERVIEW_STAGES)
async def test_the_write_moves_the_stage_only_forward(stage: str) -> None:
    execute = AsyncMock(return_value=1)
    with patch(_EXECUTE, execute):
        moved = await chat_db_module.update_chat_session_interview_stage(
            "sess-1", "user-1", stage
        )

    assert moved is True
    query, *params = execute.await_args.args
    assert params == [
        "sess-1",
        "user-1",
        stage,
        ONBOARDING_INTERVIEW_KIND,
        INTERVIEW_STAGES.index(stage) + 1,
        None,
    ]
    # Owner-scoped, interview-only, and only past a lower stage.
    assert 'WHERE "id" = $1 AND "userId" = $2' in query
    assert "\"metadata\"->>'kind' = $4" in query
    assert query.rstrip().endswith("< $5")
    for rank, known in enumerate(INTERVIEW_STAGES, start=1):
        assert f"= '{known}' THEN {rank}" in query
    # No stage yet ranks lowest; one this code does not know ranks past every
    # stage it does, so it is never overwritten.
    assert "IS NULL THEN 0" in query
    assert f"ELSE {len(INTERVIEW_STAGES) + 1} END" in query
    # Only the stage key is merged in, with the first task's when one is
    # given; null keys are stripped, so the rest of the metadata stays.
    assert (
        "jsonb_strip_nulls(jsonb_build_object('onboarding_interview_stage', "
        "$3::text, 'onboarding_interview_first_task', $6::text))"
    ) in query


@pytest.mark.asyncio
async def test_the_closing_write_stores_the_agreed_first_task_with_it() -> None:
    execute = AsyncMock(return_value=1)
    with patch(_EXECUTE, execute):
        assert await chat_db_module.update_chat_session_interview_stage(
            "sess-1", "user-1", INTERVIEW_STAGE_CLOSED, "Chase unpaid invoices"
        )

    _, *params = execute.await_args.args
    assert params[2] == INTERVIEW_STAGE_CLOSED
    assert params[5] == "Chase unpaid invoices"


@pytest.mark.asyncio
async def test_a_write_that_moved_nothing_says_so() -> None:
    with patch(_EXECUTE, AsyncMock(return_value=0)):
        assert not await chat_db_module.update_chat_session_interview_stage(
            "sess-1", "user-1", INTERVIEW_STAGE_CLOSED
        )


@pytest.mark.asyncio
async def test_an_unknown_stage_is_never_written() -> None:
    execute = AsyncMock()
    with patch(_EXECUTE, execute), pytest.raises(ValueError):
        await chat_db_module.update_chat_session_interview_stage(
            "sess-1", "user-1", "reopened"
        )

    execute.assert_not_awaited()


def _interview() -> ChatSession:
    session = ChatSession.new("user-1", dry_run=False)
    session.metadata.kind = ONBOARDING_INTERVIEW_KIND
    return session


@asynccontextmanager
async def _lock(*_args, **_kwargs):
    yield True


@pytest.mark.asyncio
async def test_a_moved_stage_is_written_through_the_cache() -> None:
    cached = _interview()
    store = MagicMock(update_chat_session_interview_stage=AsyncMock(return_value=True))
    cache = AsyncMock()
    with (
        patch.object(model, "_get_session_lock", _lock),
        patch.object(model, "chat_db", return_value=store),
        patch.object(model, "_get_session_from_cache", AsyncMock(return_value=cached)),
        patch.object(model, "cache_chat_session", cache),
    ):
        assert await model.update_session_interview_stage(
            cached.session_id, "user-1", INTERVIEW_STAGE_CLOSED
        )

    assert cache.await_args.args[0].metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_CLOSED
    )


@pytest.mark.asyncio
async def test_a_first_task_is_written_through_the_cache_with_its_stage() -> None:
    cached = _interview()
    store = MagicMock(update_chat_session_interview_stage=AsyncMock(return_value=True))
    cache = AsyncMock()
    with (
        patch.object(model, "_get_session_lock", _lock),
        patch.object(model, "chat_db", return_value=store),
        patch.object(model, "_get_session_from_cache", AsyncMock(return_value=cached)),
        patch.object(model, "cache_chat_session", cache),
    ):
        assert await model.update_session_interview_stage(
            cached.session_id,
            "user-1",
            INTERVIEW_STAGE_CLOSED,
            first_task="Chase unpaid invoices",
        )

    store.update_chat_session_interview_stage.assert_awaited_once_with(
        cached.session_id,
        "user-1",
        INTERVIEW_STAGE_CLOSED,
        first_task="Chase unpaid invoices",
    )
    written = cache.await_args.args[0].metadata
    assert written.onboarding_interview_stage == INTERVIEW_STAGE_CLOSED
    assert written.onboarding_interview_first_task == "Chase unpaid invoices"


@pytest.mark.asyncio
async def test_a_stale_turn_keeps_the_first_task_in_the_cache() -> None:
    """The end signal writes the task mid-turn; the turn's own copy, saved at
    its end, does not carry it and must not drop it."""
    stale = _interview()
    cached = stale.model_copy(deep=True)
    cached.metadata.onboarding_interview_stage = INTERVIEW_STAGE_CLOSED
    cached.metadata.onboarding_interview_first_task = "Chase unpaid invoices"
    store = MagicMock(get_next_sequence=AsyncMock(return_value=0))
    cache = AsyncMock()
    with (
        patch.object(model, "_get_session_lock", _lock),
        patch.object(model, "chat_db", return_value=store),
        patch.object(model, "_save_session_to_db", AsyncMock()),
        patch.object(model, "_get_session_from_cache", AsyncMock(return_value=cached)),
        patch.object(model, "cache_chat_session", cache),
    ):
        saved = await model.upsert_chat_session(stale)

    assert saved.metadata.onboarding_interview_first_task == "Chase unpaid invoices"


@pytest.mark.asyncio
async def test_with_no_cached_copy_the_stored_first_task_is_cached() -> None:
    stale = _interview()
    stored = ChatSessionInfo.model_validate(stale.model_dump())
    stored.metadata.onboarding_interview_stage = INTERVIEW_STAGE_CLOSED
    stored.metadata.onboarding_interview_first_task = "Chase unpaid invoices"
    store = MagicMock(
        get_next_sequence=AsyncMock(return_value=0),
        get_chat_session_metadata=AsyncMock(return_value=stored),
    )
    cache = AsyncMock()
    with (
        patch.object(model, "_get_session_lock", _lock),
        patch.object(model, "chat_db", return_value=store),
        patch.object(model, "_save_session_to_db", AsyncMock()),
        patch.object(model, "_get_session_from_cache", AsyncMock(return_value=None)),
        patch.object(model, "cache_chat_session", cache),
    ):
        await model.upsert_chat_session(stale)

    assert cache.await_args.args[0].metadata.onboarding_interview_first_task == (
        "Chase unpaid invoices"
    )


@pytest.mark.asyncio
async def test_a_stage_the_database_refused_leaves_the_cache_alone() -> None:
    store = MagicMock(update_chat_session_interview_stage=AsyncMock(return_value=False))
    cache = AsyncMock()
    with (
        patch.object(model, "_get_session_lock", _lock),
        patch.object(model, "chat_db", return_value=store),
        patch.object(model, "cache_chat_session", cache),
    ):
        assert not await model.update_session_interview_stage(
            "sess-1", "user-1", INTERVIEW_STAGE_WRAP_UP_ISSUED
        )

    cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_cache_write_evicts_the_stale_copy() -> None:
    store = MagicMock(update_chat_session_interview_stage=AsyncMock(return_value=True))
    evict = AsyncMock()
    with (
        patch.object(model, "_get_session_lock", _lock),
        patch.object(model, "chat_db", return_value=store),
        patch.object(
            model, "_get_session_from_cache", AsyncMock(return_value=_interview())
        ),
        patch.object(
            model, "cache_chat_session", AsyncMock(side_effect=RuntimeError("down"))
        ),
        patch.object(model, "invalidate_session_cache", evict),
    ):
        assert await model.update_session_interview_stage(
            "sess-1", "user-1", INTERVIEW_STAGE_HANDED_OFF
        )

    evict.assert_awaited_once_with("sess-1")


@pytest.mark.asyncio
async def test_a_stale_turn_does_not_reopen_a_closed_interview_in_the_cache() -> None:
    """A turn that started before the interview closed still holds the old
    stage; its end-of-turn save keeps the one the cache has."""
    stale = _interview()
    cached = stale.model_copy(deep=True)
    cached.metadata.onboarding_interview_stage = INTERVIEW_STAGE_CLOSED
    store = MagicMock(get_next_sequence=AsyncMock(return_value=0))
    cache = AsyncMock()
    with (
        patch.object(model, "_get_session_lock", _lock),
        patch.object(model, "chat_db", return_value=store),
        patch.object(model, "_save_session_to_db", AsyncMock()),
        patch.object(model, "_get_session_from_cache", AsyncMock(return_value=cached)),
        patch.object(model, "cache_chat_session", cache),
    ):
        saved = await model.upsert_chat_session(stale)

    assert saved.metadata.onboarding_interview_stage == INTERVIEW_STAGE_CLOSED
    assert cache.await_args.args[0].metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_CLOSED
    )


@pytest.mark.asyncio
async def test_with_no_cached_copy_the_stored_stage_is_cached() -> None:
    stale = _interview()
    stored = ChatSessionInfo.model_validate(stale.model_dump())
    stored.metadata.onboarding_interview_stage = INTERVIEW_STAGE_HANDED_OFF
    store = MagicMock(
        get_next_sequence=AsyncMock(return_value=0),
        get_chat_session_metadata=AsyncMock(return_value=stored),
    )
    cache = AsyncMock()
    with (
        patch.object(model, "_get_session_lock", _lock),
        patch.object(model, "chat_db", return_value=store),
        patch.object(model, "_save_session_to_db", AsyncMock()),
        patch.object(model, "_get_session_from_cache", AsyncMock(return_value=None)),
        patch.object(model, "cache_chat_session", cache),
    ):
        await model.upsert_chat_session(stale)

    assert cache.await_args.args[0].metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_HANDED_OFF
    )


# ---------- the ending arm ----------


@pytest.mark.asyncio
async def test_the_ending_arm_is_stamped_once_on_the_owner_s_interview() -> None:
    execute = AsyncMock(return_value=1)
    with patch(_EXECUTE, execute):
        stamped = await chat_db_module.update_chat_session_interview_ending(
            "sess-1", "user-1", "03-summary-chips"
        )

    assert stamped is True
    query, *params = execute.await_args.args
    assert params == ["sess-1", "user-1", "03-summary-chips", ONBOARDING_INTERVIEW_KIND]
    assert 'WHERE "id" = $1 AND "userId" = $2' in query
    assert "\"metadata\"->>'kind' = $4" in query
    # Never over an arm already stamped: the first one read is the one kept.
    assert query.rstrip().endswith(
        "\"metadata\"->>'onboarding_interview_ending' IS NULL"
    )
    assert "jsonb_build_object('onboarding_interview_ending', $3::text)" in query


@pytest.mark.asyncio
async def test_a_stamped_ending_is_written_through_the_cache() -> None:
    cached = _interview()
    store = MagicMock(update_chat_session_interview_ending=AsyncMock(return_value=True))
    cache = AsyncMock()
    with (
        patch.object(model, "_get_session_lock", _lock),
        patch.object(model, "chat_db", return_value=store),
        patch.object(model, "_get_session_from_cache", AsyncMock(return_value=cached)),
        patch.object(model, "cache_chat_session", cache),
    ):
        assert await model.update_session_interview_ending(
            cached.session_id, "user-1", "09-first-task"
        )

    assert cache.await_args.args[0].metadata.onboarding_interview_ending == (
        "09-first-task"
    )


@pytest.mark.asyncio
async def test_an_ending_already_stamped_leaves_the_cache_alone() -> None:
    store = MagicMock(
        update_chat_session_interview_ending=AsyncMock(return_value=False)
    )
    cache = AsyncMock()
    with (
        patch.object(model, "_get_session_lock", _lock),
        patch.object(model, "chat_db", return_value=store),
        patch.object(model, "cache_chat_session", cache),
    ):
        assert not await model.update_session_interview_ending(
            "sess-1", "user-1", "09-first-task"
        )

    cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_stale_turn_keeps_the_stamped_ending_in_the_cache() -> None:
    stale = _interview()
    cached = stale.model_copy(deep=True)
    cached.metadata.onboarding_interview_ending = "02-profile-card"
    store = MagicMock(get_next_sequence=AsyncMock(return_value=0))
    cache = AsyncMock()
    with (
        patch.object(model, "_get_session_lock", _lock),
        patch.object(model, "chat_db", return_value=store),
        patch.object(model, "_save_session_to_db", AsyncMock()),
        patch.object(model, "_get_session_from_cache", AsyncMock(return_value=cached)),
        patch.object(model, "cache_chat_session", cache),
    ):
        saved = await model.upsert_chat_session(stale)

    assert saved.metadata.onboarding_interview_ending == "02-profile-card"


# ---------- against the real database ----------


@pytest.mark.asyncio(loop_scope="session")
async def test_the_database_stores_the_first_task_only_with_the_close(
    setup_test_user, test_user_id
) -> None:
    """The close and its task are one statement: a second close is refused
    and keeps the first task, and the hand-off, which carries none, keeps it
    as well, with the rest of the metadata untouched throughout."""
    created = await chat_db_module.create_chat_session(
        str(uuid4()),
        test_user_id,
        metadata=ChatSessionMetadata(
            kind=ONBOARDING_INTERVIEW_KIND, origin="interactive"
        ),
    )
    session_id = created.session_id

    async def stored() -> ChatSessionMetadata:
        info = await chat_db_module.get_chat_session_metadata(session_id)
        assert info is not None
        return info.metadata

    assert await chat_db_module.update_chat_session_interview_stage(
        session_id, test_user_id, INTERVIEW_STAGE_CLOSED, "Chase unpaid invoices"
    )
    closed = await stored()
    assert closed.onboarding_interview_stage == INTERVIEW_STAGE_CLOSED
    assert closed.onboarding_interview_first_task == "Chase unpaid invoices"

    assert not await chat_db_module.update_chat_session_interview_stage(
        session_id, test_user_id, INTERVIEW_STAGE_CLOSED, "Post the week's specials"
    )
    assert (await stored()).onboarding_interview_first_task == "Chase unpaid invoices"

    assert await chat_db_module.update_chat_session_interview_stage(
        session_id, test_user_id, INTERVIEW_STAGE_HANDED_OFF
    )
    handed_off = await stored()
    assert handed_off.onboarding_interview_stage == INTERVIEW_STAGE_HANDED_OFF
    assert handed_off.onboarding_interview_first_task == "Chase unpaid invoices"
    assert handed_off.kind == ONBOARDING_INTERVIEW_KIND
    assert handed_off.origin == "interactive"
