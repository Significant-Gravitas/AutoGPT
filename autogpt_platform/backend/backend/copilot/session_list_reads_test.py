"""The chat list reads its page of chats and their total side by side, and the
sidebar reads its groups and its pinned chats side by side: each stand-in
answers only once both are being awaited."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture

from backend.copilot import model as chat_model
from backend.util.testing import Rendezvous


@pytest.mark.asyncio
async def test_a_sessions_page_and_its_total_are_read_side_by_side(
    mocker: MockerFixture,
) -> None:
    meeting = Rendezvous(2)
    db = SimpleNamespace(
        get_user_chat_sessions=AsyncMock(side_effect=meeting.returning(["s1", "s2"])),
        get_user_session_count=AsyncMock(side_effect=meeting.returning(7)),
    )
    mocker.patch.object(chat_model, "chat_db", return_value=db)

    sessions, total = await chat_model.get_user_sessions(
        "user-1", 2, 0, organization_id="org-1", expert_id="expert-1"
    )

    assert (sessions, total) == (["s1", "s2"], 7)
    db.get_user_chat_sessions.assert_awaited_once_with(
        "user-1",
        2,
        0,
        organization_id="org-1",
        title_contains=None,
        expert_id="expert-1",
        autopilot_only=False,
        pinned_first=True,
        pinned=None,
    )
    db.get_user_session_count.assert_awaited_once_with(
        "user-1",
        organization_id="org-1",
        expert_id="expert-1",
        autopilot_only=False,
        pinned=None,
    )


@pytest.mark.asyncio
async def test_a_page_and_its_total_apply_the_same_scope_and_pinned_filter(
    mocker: MockerFixture,
) -> None:
    db = SimpleNamespace(
        get_user_chat_sessions=AsyncMock(return_value=[]),
        get_user_session_count=AsyncMock(return_value=0),
    )
    mocker.patch.object(chat_model, "chat_db", return_value=db)

    await chat_model.get_user_sessions("user-1", autopilot_only=True, pinned=False)

    for read in (db.get_user_chat_sessions, db.get_user_session_count):
        assert read.await_args.kwargs["autopilot_only"] is True
        assert read.await_args.kwargs["pinned"] is False


@pytest.mark.asyncio
async def test_pinned_chats_are_read_beside_the_groups_and_kept_out_of_them(
    mocker: MockerFixture,
) -> None:
    meeting = Rendezvous(2)
    db = SimpleNamespace(
        get_user_chat_session_groups=AsyncMock(
            side_effect=meeting.returning(["group"])
        ),
        get_user_chat_sessions=AsyncMock(side_effect=meeting.returning(["pinned"])),
    )
    mocker.patch.object(chat_model, "chat_db", return_value=db)

    groups, pinned = await chat_model.get_user_session_groups(
        "user-1", 3, 30, organization_id="org-1", pinned_separately=True
    )

    assert (groups, pinned) == (["group"], ["pinned"])
    db.get_user_chat_session_groups.assert_awaited_once_with(
        "user-1", 3, 30, organization_id="org-1", pinned=False
    )
    assert db.get_user_chat_sessions.await_args.kwargs["pinned"] is True


@pytest.mark.asyncio
async def test_without_a_pinned_section_the_groups_keep_every_chat(
    mocker: MockerFixture,
) -> None:
    db = SimpleNamespace(
        get_user_chat_session_groups=AsyncMock(return_value=["group"]),
        get_user_chat_sessions=AsyncMock(),
    )
    mocker.patch.object(chat_model, "chat_db", return_value=db)

    groups, pinned = await chat_model.get_user_session_groups("user-1", 3, 30)

    assert (groups, pinned) == (["group"], [])
    db.get_user_chat_session_groups.assert_awaited_once_with(
        "user-1", 3, 30, organization_id=None
    )
    db.get_user_chat_sessions.assert_not_awaited()
