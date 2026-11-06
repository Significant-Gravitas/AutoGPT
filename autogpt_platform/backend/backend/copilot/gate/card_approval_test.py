"""The card answers on a real Redis: what the chat route writes, a confirm
takes once, for that user, in that chat."""

import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.gate import card_approval
from backend.copilot.gate.card_approval import record_card_decisions, take_card_answer
from backend.data import redis_client
from backend.data.redis_client import get_redis_async
from backend.util.testing import is_tcp_port_reachable

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="needs Redis",
)


async def test_an_approval_is_taken_once_by_its_user_in_its_chat():
    chat, other_chat = f"s-{uuid.uuid4()}", f"s-{uuid.uuid4()}"
    await record_card_decisions(
        "u-1",
        chat,
        "Approved: create Otto (confirmation_id: c-1).\n"
        "Not approved: do not create Ada, discard that proposal "
        "(confirmation_id: c-2).",
    )
    ttl = await (await get_redis_async()).ttl(card_approval._key(chat))

    assert 0 < ttl <= card_approval._TTL_SECONDS
    assert await take_card_answer("u-1", other_chat, "c-1") is None
    assert await take_card_answer("u-2", chat, "c-1") is None
    assert await take_card_answer("u-1", chat, "c-3") is None
    assert await take_card_answer("u-1", chat, "c-1") == "approved"
    assert await take_card_answer("u-1", chat, "c-1") is None


async def test_a_decline_is_final():
    chat = f"s-{uuid.uuid4()}"
    await _say(chat, "Approved: create Otto (confirmation_id: c-1).")
    await _say(
        chat,
        "Not approved: do not create Otto, discard that proposal "
        "(confirmation_id: c-1).",
    )
    await _say(chat, "Approved: create Otto (confirmation_id: c-1).")

    assert await take_card_answer("u-1", chat, "c-1") == "declined"
    assert await take_card_answer("u-1", chat, "c-1") == "declined"


async def test_declining_a_hires_other_choice_declines_this_one():
    chat = f"s-{uuid.uuid4()}"
    await _say(
        chat,
        "Not approved: do not hire James, discard that proposal "
        "(confirmation_id: c-plain).",
    )
    await _say(chat, "Approved: hire James (confirmation_id: c-start).")

    assert await take_card_answer("u-1", chat, "c-start", "c-plain") == "declined"


async def test_concurrent_confirms_of_one_approval_take_it_exactly_once():
    chat = f"s-{uuid.uuid4()}"
    await _say(chat, "Approved: create Otto (confirmation_id: c-1).")
    redis = await get_redis_async()
    # Both confirms read "no decline" before either takes the approval.
    both_read = asyncio.Barrier(2)

    async def hmget(*args):
        answers = await redis.hmget(*args)
        await asyncio.wait_for(both_read.wait(), timeout=5)
        return answers

    racing = AsyncMock(return_value=_Racing(redis, hmget))
    with patch.object(card_approval, "get_redis_async", racing):
        answers = await asyncio.gather(
            take_card_answer("u-1", chat, "c-1"), take_card_answer("u-1", chat, "c-1")
        )

    assert sorted(answers, key=str) == [None, "approved"]


async def _say(chat: str, line: str) -> None:
    await record_card_decisions("u-1", chat, line)


class _Racing:
    """The real client, its read held until both callers have made it."""

    def __init__(self, redis, hmget):
        self._redis = redis
        self.hmget = hmget

    def __getattr__(self, name):
        return getattr(self._redis, name)
