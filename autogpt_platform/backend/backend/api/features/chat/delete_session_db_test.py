"""Deleting a chat whose turn is running or queued, against a real database.

A turn's slot is its chat row's ``chatStatus``, so the account's running and
queued counts are counts of those rows.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from autogpt_libs.auth.models import RequestContext
from fastapi import HTTPException
from prisma.models import ChatSession, User
from pytest_mock import MockerFixture

from backend.api.features.chat import routes as chat_routes
from backend.api.features.chat.routes import CancelSessionResponse, delete_session
from backend.copilot.model import (
    CHAT_STATUS_IDLE,
    CHAT_STATUS_QUEUED,
    CHAT_STATUS_RUNNING,
    create_chat_session,
)
from backend.copilot.turn_queue import count_inflight_turns
from backend.data.db_accessors import chat_db
from backend.util.test import SpinTestServer

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.mark.parametrize("status", [CHAT_STATUS_RUNNING, CHAT_STATUS_QUEUED])
@pytest.mark.parametrize("turn_stops", [True, False], ids=["stopped", "still-busy"])
async def test_deleting_a_busy_chat_stops_its_turn_and_keeps_the_count(
    server: SpinTestServer, mocker: MockerFixture, status: str, turn_stops: bool
) -> None:
    """``still-busy``: the stop is not confirmed, or a new turn was admitted
    meanwhile, so the chat keeps its slot and the delete is refused."""
    user_id = str(uuid4())
    await User.prisma().create(data={"id": user_id, "email": f"{user_id}@example.com"})
    try:
        chat = await create_chat_session(user_id, dry_run=False)
        assert await chat_db().update_chat_session_status(
            session_id=chat.session_id,
            expect_status=CHAT_STATUS_IDLE,
            status=status,
            user_id=user_id,
        )

        async def stop_turn(session_id: str, user_id: str) -> CancelSessionResponse:
            if turn_stops:
                await chat_db().update_chat_session_status(
                    session_id=session_id,
                    expect_status=status,
                    status=CHAT_STATUS_IDLE,
                    user_id=user_id,
                )
            return CancelSessionResponse(cancelled=True)

        stop = mocker.patch.object(
            chat_routes, "cancel_session_task", side_effect=stop_turn
        )
        mocker.patch.object(
            chat_routes,
            "ChatConfig",
            return_value=SimpleNamespace(e2b_active=False, e2b_api_key=None),
        )

        if turn_stops:
            response = await delete_session(chat.session_id, user_id, _ctx(user_id))
            assert response.status_code == 204
        else:
            with pytest.raises(HTTPException) as refused:
                await delete_session(chat.session_id, user_id, _ctx(user_id))
            assert refused.value.status_code == 409

        stop.assert_awaited_once_with(chat.session_id, user_id)
        assert await count_inflight_turns(user_id) == (0 if turn_stops else 1)
        row = await ChatSession.prisma().find_unique(where={"id": chat.session_id})
        assert (row is None) == turn_stops
    finally:
        await User.prisma().delete(where={"id": user_id})


async def test_deleting_an_idle_chat_stops_nothing(
    server: SpinTestServer, mocker: MockerFixture
) -> None:
    user_id = str(uuid4())
    await User.prisma().create(data={"id": user_id, "email": f"{user_id}@example.com"})
    try:
        chat = await create_chat_session(user_id, dry_run=False)
        stop = mocker.patch.object(chat_routes, "cancel_session_task")
        mocker.patch.object(
            chat_routes,
            "ChatConfig",
            return_value=SimpleNamespace(e2b_active=False, e2b_api_key=None),
        )

        response = await delete_session(chat.session_id, user_id, _ctx(user_id))

        assert response.status_code == 204
        stop.assert_not_called()
        assert (
            await ChatSession.prisma().find_unique(where={"id": chat.session_id})
            is None
        )
    finally:
        await User.prisma().delete(where={"id": user_id})


def _ctx(user_id: str) -> RequestContext:
    return RequestContext(
        user_id=user_id,
        org_id=str(uuid4()),
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=True,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )
