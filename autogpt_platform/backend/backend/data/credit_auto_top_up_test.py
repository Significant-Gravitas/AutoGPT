from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from prisma.enums import CreditTransactionType
from prisma.models import CreditTransaction, User

from backend.data import credit
from backend.data.credit import UsageTransactionMetadata, UserCredit


@pytest.fixture
def top_ups(mocker):
    mocker.patch.object(
        credit,
        "get_auto_top_up",
        AsyncMock(return_value=MagicMock(threshold=100, amount=500)),
    )
    mocker.patch.object(
        UserCredit, "_add_transaction", AsyncMock(return_value=(10, "key"))
    )
    top_up = AsyncMock()
    mocker.patch.object(UserCredit, "_top_up_credits", top_up)
    return top_up


async def _spend(metadata: UsageTransactionMetadata) -> None:
    await UserCredit().spend_credits(user_id="u1", cost=5, metadata=metadata)


@pytest.mark.asyncio
async def test_each_chat_gets_its_own_auto_top_up(top_ups):
    await _spend(UsageTransactionMetadata(chat_session_id="chat-1"))
    await _spend(UsageTransactionMetadata(chat_session_id="chat-2"))
    await _spend(UsageTransactionMetadata(graph_exec_id="run-1", graph_id="g"))

    keys = [call.kwargs["key"] for call in top_ups.await_args_list]
    assert keys == [
        "AUTO-TOP-UP-u1-chat-1",
        "AUTO-TOP-UP-u1-chat-2",
        "AUTO-TOP-UP-u1-run-1",
    ]


@pytest.mark.asyncio
async def test_usage_outside_a_run_or_chat_still_has_a_key(top_ups):
    # Without one, every spend below the threshold could start another charge
    # while the first top-up is still inactive.
    await _spend(UsageTransactionMetadata(reason="CoPilot daily rate limit reset"))

    assert top_ups.await_args.kwargs["key"] == "AUTO-TOP-UP-u1-None"


@pytest.mark.asyncio(loop_scope="session")
async def test_a_chat_with_an_uncharged_legacy_top_up_is_not_charged_again(
    server, top_ups
):
    # Before chats had their own id, a chat's top-up was keyed on
    # copilot-session-<id>; only a failed or pending one keeps that key.
    user_id = f"top-up-{uuid4()}"
    await User.prisma().create(data={"id": user_id, "email": f"{user_id}@example.com"})
    try:
        await CreditTransaction.prisma().create(
            data={
                "userId": user_id,
                "transactionKey": f"AUTO-TOP-UP-{user_id}-copilot-session-chat-1",
                "amount": 500,
                "type": CreditTransactionType.TOP_UP,
                "isActive": False,
            }
        )
        spend = UserCredit().spend_credits

        await spend(user_id=user_id, cost=5, metadata=_chat("chat-1"))
        top_ups.assert_not_awaited()

        await spend(user_id=user_id, cost=5, metadata=_chat("chat-2"))
        assert top_ups.await_args.kwargs["key"] == f"AUTO-TOP-UP-{user_id}-chat-2"
    finally:
        await CreditTransaction.prisma().delete_many(where={"userId": user_id})
        await User.prisma().delete_many(where={"id": user_id})


def _chat(chat_session_id: str) -> UsageTransactionMetadata:
    return UsageTransactionMetadata(chat_session_id=chat_session_id)
