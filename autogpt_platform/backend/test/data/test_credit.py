from datetime import datetime, timezone

import pytest
from prisma.enums import CreditTransactionType
from prisma.models import CreditTransaction

from backend.blocks.llm import AITextGeneratorBlock
from backend.data.credit import BetaUserCredit
from backend.data.execution import NodeExecutionEntry
from backend.data.user import DEFAULT_USER_ID
from backend.integrations.credentials_store import openai_credentials
from backend.util.test import SpinTestServer

REFILL_VALUE = 1000
user_credit = BetaUserCredit(REFILL_VALUE)


async def disable_test_user_transactions():
    await CreditTransaction.prisma().delete_many(where={"userId": DEFAULT_USER_ID})


async def top_up(amount: int):
    await user_credit._add_transaction(
        DEFAULT_USER_ID,
        amount,
        CreditTransactionType.TOP_UP,
    )


@pytest.mark.asyncio(scope="session")
async def test_block_credit_usage(server: SpinTestServer):
    await disable_test_user_transactions()
    await top_up(100)
    current_credit = await user_credit.get_credits(DEFAULT_USER_ID)

    spending_amount_1 = await user_credit.spend_credits(
        NodeExecutionEntry(
            user_id=DEFAULT_USER_ID,
            graph_id="test_graph",
            node_id="test_node",
            graph_exec_id="test_graph_exec",
            node_exec_id="test_node_exec",
            block_id=AITextGeneratorBlock().id,
            data={
                "model": "gpt-4-turbo",
                "credentials": {
                    "id": openai_credentials.id,
                    "provider": openai_credentials.provider,
                    "type": openai_credentials.type,
                },
            },
        ),
        0.0,
        0.0,
    )
    assert spending_amount_1 > 0

    spending_amount_2 = await user_credit.spend_credits(
        NodeExecutionEntry(
            user_id=DEFAULT_USER_ID,
            graph_id="test_graph",
            node_id="test_node",
            graph_exec_id="test_graph_exec",
            node_exec_id="test_node_exec",
            block_id=AITextGeneratorBlock().id,
            data={"model": "gpt-4-turbo", "api_key": "owned_api_key"},
        ),
        0.0,
        0.0,
    )
    assert spending_amount_2 == 0

    new_credit = await user_credit.get_credits(DEFAULT_USER_ID)
    assert new_credit == current_credit - spending_amount_1 - spending_amount_2


@pytest.mark.asyncio(scope="session")
async def test_block_credit_top_up(server: SpinTestServer):
    await disable_test_user_transactions()
    current_credit = await user_credit.get_credits(DEFAULT_USER_ID)

    await top_up(100)

    new_credit = await user_credit.get_credits(DEFAULT_USER_ID)
    assert new_credit == current_credit + 100


@pytest.mark.asyncio(scope="session")
async def test_block_credit_reset(server: SpinTestServer):
    await disable_test_user_transactions()
    month1 = 1
    month2 = 2

    # set the calendar to month 2 but use current time from now
    user_credit.time_now = lambda: datetime.now(timezone.utc).replace(
        month=month2, day=1
    )
    month2credit = await user_credit.get_credits(DEFAULT_USER_ID)

    # Month 1 result should only affect month 1
    user_credit.time_now = lambda: datetime.now(timezone.utc).replace(
        month=month1, day=1
    )
    month1credit = await user_credit.get_credits(DEFAULT_USER_ID)
    await top_up(100)
    assert await user_credit.get_credits(DEFAULT_USER_ID) == month1credit + 100

    # Month 2 balance is unaffected
    user_credit.time_now = lambda: datetime.now(timezone.utc).replace(
        month=month2, day=1
    )
    assert await user_credit.get_credits(DEFAULT_USER_ID) == month2credit


@pytest.mark.asyncio(scope="session")
async def test_credit_refill(server: SpinTestServer):
    await disable_test_user_transactions()
    balance = await user_credit.get_credits(DEFAULT_USER_ID)
    assert balance == REFILL_VALUE
