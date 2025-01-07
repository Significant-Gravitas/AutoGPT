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


async def disable_test_user_transactions():
    await CreditTransaction.prisma().delete_many(where={"userId": DEFAULT_USER_ID})


@pytest.mark.asyncio(scope="session")
async def test_block_credit_usage(server: SpinTestServer):
    await disable_test_user_transactions()
    await user_credit.top_up_credits(DEFAULT_USER_ID, 100)
    current_credit = await user_credit.get_credits(DEFAULT_USER_ID)

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

    new_credit = await user_credit.get_credits(DEFAULT_USER_ID)
    assert new_credit == current_credit - spending_amount_1 - spending_amount_2


@pytest.mark.asyncio(scope="session")
async def test_block_credit_top_up(server: SpinTestServer):
    await disable_test_user_transactions()
    current_credit = await user_credit.get_credits(DEFAULT_USER_ID)

    await user_credit.top_up_credits(DEFAULT_USER_ID, 100)

    new_credit = await user_credit.get_credits(DEFAULT_USER_ID)
    assert new_credit == current_credit + 100


@pytest.mark.asyncio(scope="session")
async def test_block_credit_reset(server: SpinTestServer):
    await disable_test_user_transactions()
    month1 = 1
    month2 = 2

    # set the calendar to month 2 but use current time from now
    user_credit.time_now = lambda: datetime.now().replace(month=month2)
    month2credit = await user_credit.get_credits(DEFAULT_USER_ID)

    # Month 1 result should only affect month 1
    user_credit.time_now = lambda: datetime.now().replace(month=month1)
    month1credit = await user_credit.get_credits(DEFAULT_USER_ID)
    await user_credit.top_up_credits(DEFAULT_USER_ID, 100)
    assert await user_credit.get_credits(DEFAULT_USER_ID) == month1credit + 100

    # Month 2 balance is unaffected
    user_credit.time_now = lambda: datetime.now().replace(month=month2)
    assert await user_credit.get_credits(DEFAULT_USER_ID) == month2credit


@pytest.mark.asyncio(scope="session")
async def test_credit_refill(server: SpinTestServer):
    await disable_test_user_transactions()
    balance = await user_credit.get_credits(DEFAULT_USER_ID)
    assert balance == REFILL_VALUE
