import pytest

from autogpt_server.blocks.llm import AITextGeneratorBlock
from autogpt_server.data.credit import UserCredit
from autogpt_server.data.user import DEFAULT_USER_ID
from autogpt_server.util.test import SpinTestServer

user_credit = UserCredit(0)


@pytest.mark.asyncio(scope="session")
async def test_block_credit_usage(server: SpinTestServer):
    current_credit = await user_credit.get_or_refill_credit(DEFAULT_USER_ID)

    spending_amount_1 = await user_credit.spend_credits(
        DEFAULT_USER_ID,
        current_credit,
        AITextGeneratorBlock(),
        {"model": "gpt-4-turbo"},
        0.0,
        0.0,
        validate_balance=False,
    )
    assert spending_amount_1 > 0

    spending_amount_2 = await user_credit.spend_credits(
        DEFAULT_USER_ID,
        current_credit,
        AITextGeneratorBlock(),
        {"model": "gpt-4-turbo", "api_key": "owned_api_key"},
        0.0,
        0.0,
        validate_balance=False,
    )
    assert spending_amount_2 == 0

    new_credit = await user_credit.get_or_refill_credit(DEFAULT_USER_ID)
    assert new_credit == current_credit - spending_amount_1 - spending_amount_2


@pytest.mark.asyncio(scope="session")
async def test_block_credit_top_up(server: SpinTestServer):
    current_credit = await user_credit.get_or_refill_credit(DEFAULT_USER_ID)

    await user_credit.top_up_credits(DEFAULT_USER_ID, 100)

    new_credit = await user_credit.get_or_refill_credit(DEFAULT_USER_ID)
    assert new_credit == current_credit + 100
