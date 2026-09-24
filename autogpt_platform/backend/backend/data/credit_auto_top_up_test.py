from unittest.mock import AsyncMock, MagicMock

import pytest

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
