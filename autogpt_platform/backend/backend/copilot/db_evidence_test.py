import pytest

from . import db_test
from .db import get_chat_messages_paginated
from .model import ChatMessage, ChatSession, upsert_chat_session

mock_db = db_test.mock_db


@pytest.mark.asyncio
@pytest.mark.parametrize("bounded", [False, True])
async def test_targeted_messages_keep_ownership_revision_and_no_expansion(
    mock_db, bounded
):
    query, expand = mock_db
    query.return_value = db_test._make_session(
        messages=[
            db_test._make_msg(142, role="tool"),
            db_test._make_msg(2, role="tool"),
        ]
    )
    page = await get_chat_messages_paginated(
        "sess-1",
        limit=2,
        user_id="user-1",
        after_sequence=0 if bounded else None,
        before_sequence=143 if bounded else None,
        message_sequences=[2, 142],
    )
    assert [m.sequence for m in page.messages] == [2, 142]
    args = query.await_args.kwargs
    assert args["where"] == {"id": "sess-1", "userId": "user-1"}
    expected = {"gte": 0, "lt": 143, "in": [2, 142]} if bounded else {"in": [2, 142]}
    assert args["include"]["Messages"]["where"]["sequence"] == expected
    expand.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_sparse_message_selection_on_owned_database_session(
    setup_test_user, test_user_id
):
    session = ChatSession.new(test_user_id, dry_run=False)
    session.messages = [
        ChatMessage(role="tool", content=f"result {n}") for n in range(200)
    ]
    saved = await upsert_chat_session(session)
    selected = await get_chat_messages_paginated(
        saved.session_id,
        limit=10,
        user_id=test_user_id,
        after_sequence=0,
        before_sequence=190,
        message_sequences=[2, 145, 190],
    )
    assert selected is not None
    assert [m.sequence for m in selected.messages] == [2, 145]
    assert [m.content for m in selected.messages] == ["result 2", "result 145"]
    assert not selected.has_more
    assert (
        await get_chat_messages_paginated(
            saved.session_id, user_id="another-owner", message_sequences=[2, 145]
        )
        is None
    )
