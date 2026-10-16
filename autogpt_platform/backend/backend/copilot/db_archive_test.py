from uuid import uuid4

import pytest
from prisma.models import User

from backend.copilot.db import (
    add_chat_message,
    admit_chat_session_turn,
    create_chat_session,
    get_chat_session_metadata,
    get_last_message_texts,
    get_user_chat_sessions,
    get_user_session_count,
    set_chat_sessions_archived,
)


@pytest.mark.asyncio(loop_scope="session")
async def test_archive_preserves_history_and_resumes_on_next_turn(setup_test_user):
    user_id = setup_test_user
    session = await create_chat_session(str(uuid4()), user_id)
    other_user = await User.prisma().create(
        data={"id": str(uuid4()), "email": f"archive-{uuid4()}@example.com"}
    )
    other = await create_chat_session(str(uuid4()), other_user.id)
    count = await get_user_session_count(user_id)

    assert await set_chat_sessions_archived([], user_id, True) == 0
    assert (
        await set_chat_sessions_archived(
            [session.session_id, other.session_id], user_id, True
        )
        == 1
    )
    assert await set_chat_sessions_archived([session.session_id], user_id, True) == 0
    assert await get_user_session_count(user_id) == count - 1
    assert await get_user_session_count(user_id, include_archived=True) == count
    assert session.session_id not in {
        s.session_id for s in await get_user_chat_sessions(user_id, limit=10_000)
    }
    assert session.session_id in {
        s.session_id
        for s in await get_user_chat_sessions(
            user_id, limit=10_000, include_archived=True
        )
    }
    assert other.session_id in {
        s.session_id for s in await get_user_chat_sessions(other_user.id)
    }
    archived = await get_chat_session_metadata(session.session_id)
    assert archived is not None and archived.updated_at == session.updated_at
    assert await set_chat_sessions_archived([session.session_id], user_id, False) == 1
    assert await set_chat_sessions_archived([session.session_id], user_id, False) == 0
    assert await get_user_session_count(user_id) == count

    await set_chat_sessions_archived([session.session_id], user_id, True)
    assert (
        await admit_chat_session_turn(
            session_id=session.session_id,
            user_id=user_id,
            expect_status=session.chat_status,
            capacity=1000,
        )
        == "admitted"
    )
    assert session.session_id in {
        s.session_id for s in await get_user_chat_sessions(user_id, limit=10_000)
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_previews_skip_hidden_empty_and_tool_messages(setup_test_user):
    session = await create_chat_session(str(uuid4()), setup_test_user)
    empty = await create_chat_session(str(uuid4()), setup_test_user)
    for sequence, (role, content, metadata) in enumerate(
        [
            ("user", "Visible question", None),
            ("assistant", "Visible answer", None),
            ("tool", "Internal output", None),
            ("user", "Kickoff context", {"hidden": True}),
            ("assistant", "  ", None),
        ]
    ):
        await add_chat_message(
            session.session_id, role, sequence, content, metadata=metadata
        )

    assert await get_last_message_texts([]) == {}
    assert await get_last_message_texts([session.session_id, empty.session_id]) == {
        session.session_id: "Visible answer"
    }
