from unittest.mock import AsyncMock

import pytest

from backend.copilot import pending_message_helpers
from backend.copilot.gate.held import HeldResult
from backend.copilot.model import ChatSession
from backend.copilot.pending_messages import (
    PendingMessage,
    drain_and_format_for_injection,
    drain_pending_for_persist,
    drain_pending_messages,
    peek_pending_messages,
    push_pending_message,
    stash_pending_for_persist,
)
from backend.copilot.pending_messages_test import fake_redis
from backend.copilot.tool_image import ToolImage

__all__ = ["fake_redis"]


@pytest.fixture
def held_image():
    return HeldResult(
        content="Approved image: " + "x" * 40_000,
        metadata={"held_call": {"review_id": "image-review"}},
        model_image=ToolImage(mime_type="image/png", data_base64="cGl4ZWxz"),
    )


async def test_text_hook_preserves_large_held_image_and_later_rows(
    fake_redis, held_image
):
    before = PendingMessage(content="before")
    after = PendingMessage(content="after")
    for row in [before, held_image, after]:
        await push_pending_message("images", row)

    injected = await drain_and_format_for_injection("images", log_prefix="test")

    assert "before" in injected
    assert "Approved image:" not in injected
    assert await drain_pending_for_persist("images") == [before]
    assert await drain_pending_messages("images") == [held_image, after]


@pytest.mark.parametrize("queue", ["primary", "persist"])
async def test_large_held_image_survives_each_queue_reader(
    fake_redis, held_image, queue
):
    if queue == "primary":
        await push_pending_message("images", held_image)
        assert await peek_pending_messages("images") == [held_image]
        assert await drain_pending_messages("images") == [held_image]
    else:
        await stash_pending_for_persist("images", [held_image])
        assert await drain_pending_for_persist("images") == [held_image]


async def test_unfolded_image_batch_is_requeued_once_in_order(
    fake_redis, held_image, monkeypatch
):
    session = ChatSession.new("user-1", dry_run=False)
    pending = [
        PendingMessage(content="before"),
        held_image,
        PendingMessage(content="after"),
    ]
    expected = list(pending)

    async def number_rows(saved_session, _log_prefix):
        for index, row in enumerate(saved_session.messages):
            row.sequence = index
        return saved_session

    monkeypatch.setattr(pending_message_helpers, "persist_session_safe", number_rows)
    await pending_message_helpers.persist_unfolded_rows(
        session, pending, log_prefix="test"
    )
    await pending_message_helpers.persist_unfolded_rows(
        session, pending, log_prefix="test"
    )

    assert not pending
    assert await drain_pending_messages(session.session_id) == expected
    assert not session.messages


async def test_unfolded_text_still_persists_without_requeue(monkeypatch):
    session = ChatSession.new("user-1", dry_run=False)
    message = PendingMessage(content="Approved result")
    pending = [message]
    persist = AsyncMock(return_value=True)
    requeue = AsyncMock()
    monkeypatch.setattr(
        pending_message_helpers, "persist_pending_as_user_rows", persist
    )
    monkeypatch.setattr(pending_message_helpers, "push_pending_message", requeue)

    await pending_message_helpers.persist_unfolded_rows(
        session, pending, log_prefix="test"
    )

    persist.assert_awaited_once_with(session, None, [message], log_prefix="test")
    requeue.assert_not_awaited()
    assert not pending
