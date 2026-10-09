"""The running cap against the database: what the spawn tools persist is
what the cap reads."""

import uuid
from unittest.mock import patch

import pytest
import pytest_asyncio
from prisma.models import User

from backend.copilot import active_turns
from backend.copilot.active_turns import acquire_turn_slot
from backend.copilot.model import (
    CHAT_STATUS_IDLE,
    CHAT_STATUS_RUNNING,
    create_chat_session,
)
from backend.data.db import prisma as db_client
from backend.data.db_accessors import chat_db
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_another_chats_sub_sessions_leave_the_users_next_message_admitted():
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"active-turns-{user_id}@example.com"}
    )
    try:
        spawner = await create_chat_session(user_id, dry_run=False)
        subs = [
            await create_chat_session(
                user_id,
                dry_run=False,
                origin="automation",
                delegated_by_session_id=spawner.session_id,
            )
            for _ in range(6)
        ]
        for session in [spawner, *subs]:
            assert await chat_db().update_chat_session_status(
                session_id=session.session_id,
                expect_status=CHAT_STATUS_IDLE,
                status=CHAT_STATUS_RUNNING,
                user_id=user_id,
            )
        chat = await create_chat_session(user_id, dry_run=False)

        with patch.object(active_turns, "get_running_turn_limit", return_value=5):
            async with acquire_turn_slot(user_id, chat.session_id) as slot:
                assert slot.admitted
    finally:
        await User.prisma().delete(where={"id": user_id})


@pytest_asyncio.fixture(autouse=True)
async def absorb_a_stale_event_loop(server: SpinTestServer):
    """An earlier test can leave the shared Prisma client bound to a closed
    loop; only the first query on the new loop fails. Spend it here."""
    try:
        await db_client.execute_raw("SELECT 1")
    except RuntimeError as error:
        if "Event loop is closed" not in str(error):
            raise
