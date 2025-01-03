from datetime import datetime

import pytest
from prisma.models import CreditTransaction

from backend.blocks.llm import AITextGeneratorBlock
from backend.data.credit import BetaUserCredit
from backend.data.user import DEFAULT_USER_ID
from backend.integrations.credentials_store import openai_credentials
from backend.util.test import SpinTestServer

REFILL_VALUE = 1000
user_credit = BetaUserCredit(REFILL_VALUE)


@pytest.mark.asyncio(scope="session")
async def test_block_credit_usage(server: SpinTestServer):
    current_credit = await user_credit.get_balance(DEFAULT_USER_ID)

    spending_amount_1 = await user_credit.spend_credits(
        DEFAULT_USER_ID,
        AITextGeneratorBlock().id,
        {
            "model": "gpt-4-turbo",
            "credentials": {
                "id": openai_credentials.id,
                "provider": openai_credentials.provider,
                "type": openai_credentials.type,
            },
        },
        0.0,
        0.0,
    )
    assert spending_amount_1 > 0

    spending_amount_2 = await user_credit.spend_credits(
        DEFAULT_USER_ID,
        AITextGeneratorBlock().id,
        {"model": "gpt-4-turbo", "api_key": "owned_api_key"},
        0.0,
        0.0,
    )
    assert spending_amount_2 == 0

    new_credit = await user_credit.get_balance(DEFAULT_USER_ID)
    assert new_credit == current_credit - spending_amount_1 - spending_amount_2


@pytest.mark.asyncio(scope="session")
async def test_block_credit_top_up(server: SpinTestServer):
    current_credit = await user_credit.get_balance(DEFAULT_USER_ID)

    await user_credit.top_up_credits(DEFAULT_USER_ID, 100)

    new_credit = await user_credit.get_balance(DEFAULT_USER_ID)
    assert new_credit == current_credit + 100


@pytest.mark.asyncio(scope="session")
async def test_block_credit_reset(server: SpinTestServer):
    old_month = datetime(2022, 1, 15)

    # Track the balance of the old month and the current month.
    user_credit.time_now = lambda: old_month
    old_month_credit = await user_credit.get_balance(DEFAULT_USER_ID)
    user_credit.time_now = lambda: datetime.now()
    cur_month_credit = await user_credit.get_balance(DEFAULT_USER_ID)

    # Cur month credit is updated upon top-up.
    await user_credit.top_up_credits(DEFAULT_USER_ID, 100)
    assert await user_credit.get_balance(DEFAULT_USER_ID) == cur_month_credit + 100

    # Old month is unaffected by the top-up.
    user_credit.time_now = lambda: old_month
    assert await user_credit.get_balance(DEFAULT_USER_ID) == old_month_credit


@pytest.mark.asyncio(scope="session")
async def test_credit_refill(server: SpinTestServer):
    # Clear all transactions within the month
    await CreditTransaction.prisma().update_many(
        where={
            "userId": DEFAULT_USER_ID,
            "createdAt": {
                "gte": datetime(2022, 2, 1),
                "lt": datetime(2022, 3, 1),
            },
        },
        data={"isActive": False},
    )
    user_credit.time_now = lambda: datetime(2022, 2, 15)

    balance = await user_credit.get_balance(DEFAULT_USER_ID)
    assert balance == REFILL_VALUE
