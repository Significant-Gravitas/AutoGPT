"""The card-approval record on a real Redis: what the chat route writes, the
gate reads once, for that user, in that chat."""

import uuid

import pytest

from backend.copilot.gate import card_approval
from backend.copilot.gate.card_approval import approved_on_card, record_card_decisions
from backend.data import redis_client
from backend.data.redis_client import get_redis_async
from backend.util.testing import is_tcp_port_reachable

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="needs Redis",
)
_CONFIRM = "confirm_expert_change"


async def test_an_approval_is_read_once_by_its_user_in_its_chat():
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
    assert not await approved_on_card(
        _CONFIRM, {"confirmation_id": "c-1"}, "u-1", other_chat
    )
    assert not await approved_on_card(_CONFIRM, {"confirmation_id": "c-1"}, "u-2", chat)
    assert not await approved_on_card(_CONFIRM, {"confirmation_id": "c-2"}, "u-1", chat)
    assert await approved_on_card(_CONFIRM, {"confirmation_id": "c-1"}, "u-1", chat)
    assert not await approved_on_card(_CONFIRM, {"confirmation_id": "c-1"}, "u-1", chat)


async def test_a_later_decline_withdraws_an_approval():
    chat = f"s-{uuid.uuid4()}"
    await record_card_decisions(
        "u-1", chat, "Approved: create Otto (confirmation_id: c-1)."
    )
    await record_card_decisions(
        "u-1",
        chat,
        "Not approved: do not create Otto, discard that proposal "
        "(confirmation_id: c-1).",
    )

    assert not await approved_on_card(_CONFIRM, {"confirmation_id": "c-1"}, "u-1", chat)
