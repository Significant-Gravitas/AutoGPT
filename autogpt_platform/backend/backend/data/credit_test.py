from datetime import datetime, timedelta, timezone

import pytest
from prisma.enums import CreditTransactionType
from prisma.models import CreditTransaction, UserBalance

from backend.blocks.llm import AITextGeneratorBlock
from backend.data.block import get_block
from backend.data.credit import BetaUserCredit, UsageTransactionMetadata
from backend.data.execution import ExecutionContext, NodeExecutionEntry
from backend.data.user import DEFAULT_USER_ID
from backend.executor.utils import block_usage_cost
from backend.integrations.credentials_store import openai_credentials
from backend.util.test import SpinTestServer

REFILL_VALUE = 1000
user_credit = BetaUserCredit(REFILL_VALUE)


async def disable_test_user_transactions():
    await CreditTransaction.prisma().delete_many(where={"userId": DEFAULT_USER_ID})
    # Also reset the balance to 0 and set updatedAt to old date to trigger monthly refill
    old_date = datetime.now(timezone.utc) - timedelta(days=35)  # More than a month ago
    await UserBalance.prisma().upsert(
        where={"userId": DEFAULT_USER_ID},
        data={
            "create": {"userId": DEFAULT_USER_ID, "balance": 0},
            "update": {"balance": 0, "updatedAt": old_date},
        },
    )


async def top_up(amount: int):
    balance, _ = await user_credit._add_transaction(
        DEFAULT_USER_ID,
        amount,
        CreditTransactionType.TOP_UP,
    )
    return balance


async def spend_credits(entry: NodeExecutionEntry) -> int:
    block = get_block(entry.block_id)
    if not block:
        raise RuntimeError(f"Block {entry.block_id} not found")

    cost, matching_filter = block_usage_cost(block=block, input_data=entry.inputs)
    await user_credit.spend_credits(
        entry.user_id,
        cost,
        UsageTransactionMetadata(
            graph_exec_id=entry.graph_exec_id,
            graph_id=entry.graph_id,
            node_id=entry.node_id,
            node_exec_id=entry.node_exec_id,
            block_id=entry.block_id,
            block=entry.block_id,
            input=matching_filter,
            reason=f"Ran block {entry.block_id} {block.name}",
        ),
    )

    return cost


@pytest.mark.asyncio(loop_scope="session")
async def test_block_credit_usage(server: SpinTestServer):
    await disable_test_user_transactions()
    await top_up(100)
    current_credit = await user_credit.get_credits(DEFAULT_USER_ID)

    spending_amount_1 = await spend_credits(
        NodeExecutionEntry(
            user_id=DEFAULT_USER_ID,
            graph_id="test_graph",
            graph_version=1,
            node_id="test_node",
            graph_exec_id="test_graph_exec",
            node_exec_id="test_node_exec",
            block_id=AITextGeneratorBlock().id,
            inputs={
                "model": "gpt-4-turbo",
                "credentials": {
                    "id": openai_credentials.id,
                    "provider": openai_credentials.provider,
                    "type": openai_credentials.type,
                },
            },
            execution_context=ExecutionContext(user_timezone="UTC"),
        ),
    )
    assert spending_amount_1 > 0

    spending_amount_2 = await spend_credits(
        NodeExecutionEntry(
            user_id=DEFAULT_USER_ID,
            graph_id="test_graph",
            graph_version=1,
            node_id="test_node",
            graph_exec_id="test_graph_exec",
            node_exec_id="test_node_exec",
            block_id=AITextGeneratorBlock().id,
            inputs={"model": "gpt-4-turbo", "api_key": "owned_api_key"},
            execution_context=ExecutionContext(user_timezone="UTC"),
        ),
    )
    assert spending_amount_2 == 0

    new_credit = await user_credit.get_credits(DEFAULT_USER_ID)
    assert new_credit == current_credit - spending_amount_1 - spending_amount_2


@pytest.mark.asyncio(loop_scope="session")
async def test_block_credit_top_up(server: SpinTestServer):
    await disable_test_user_transactions()
    current_credit = await user_credit.get_credits(DEFAULT_USER_ID)

    await top_up(100)

    new_credit = await user_credit.get_credits(DEFAULT_USER_ID)
    assert new_credit == current_credit + 100


@pytest.mark.asyncio(loop_scope="session")
async def test_block_credit_reset(server: SpinTestServer):
    """Test that BetaUserCredit provides monthly refills correctly."""
    await disable_test_user_transactions()

    # Save original time_now function for restoration
    original_time_now = user_credit.time_now

    try:
        # Test month 1 behavior
        month1 = datetime.now(timezone.utc).replace(month=1, day=1)
        user_credit.time_now = lambda: month1

        # First call in month 1 should trigger refill
        balance = await user_credit.get_credits(DEFAULT_USER_ID)
        assert balance == REFILL_VALUE  # Should get 1000 credits

        # Manually create a transaction with month 1 timestamp to establish history
        await CreditTransaction.prisma().create(
            data={
                "userId": DEFAULT_USER_ID,
                "amount": 100,
                "type": CreditTransactionType.TOP_UP,
                "runningBalance": 1100,
                "isActive": True,
                "createdAt": month1,  # Set specific timestamp
            }
        )

        # Update user balance to match
        await UserBalance.prisma().upsert(
            where={"userId": DEFAULT_USER_ID},
            data={
                "create": {"userId": DEFAULT_USER_ID, "balance": 1100},
                "update": {"balance": 1100},
            },
        )

        # Now test month 2 behavior
        month2 = datetime.now(timezone.utc).replace(month=2, day=1)
        user_credit.time_now = lambda: month2

        # In month 2, since balance (1100) > refill (1000), no refill should happen
        month2_balance = await user_credit.get_credits(DEFAULT_USER_ID)
        assert month2_balance == 1100  # Balance persists, no reset

        # Now test the refill behavior when balance is low
        # Set balance below refill threshold
        await UserBalance.prisma().update(
            where={"userId": DEFAULT_USER_ID}, data={"balance": 400}
        )

        # Create a month 2 transaction to update the last transaction time
        await CreditTransaction.prisma().create(
            data={
                "userId": DEFAULT_USER_ID,
                "amount": -700,  # Spent 700 to get to 400
                "type": CreditTransactionType.USAGE,
                "runningBalance": 400,
                "isActive": True,
                "createdAt": month2,
            }
        )

        # Move to month 3
        month3 = datetime.now(timezone.utc).replace(month=3, day=1)
        user_credit.time_now = lambda: month3

        # Should get refilled since balance (400) < refill value (1000)
        month3_balance = await user_credit.get_credits(DEFAULT_USER_ID)
        assert month3_balance == REFILL_VALUE  # Should be refilled to 1000

        # Verify the refill transaction was created
        refill_tx = await CreditTransaction.prisma().find_first(
            where={
                "userId": DEFAULT_USER_ID,
                "type": CreditTransactionType.GRANT,
                "transactionKey": {"contains": "MONTHLY-CREDIT-TOP-UP"},
            },
            order={"createdAt": "desc"},
        )
        assert refill_tx is not None, "Monthly refill transaction should be created"
        assert refill_tx.amount == 600, "Refill should be 600 (1000 - 400)"
    finally:
        # Restore original time_now function
        user_credit.time_now = original_time_now


@pytest.mark.asyncio(loop_scope="session")
async def test_credit_refill(server: SpinTestServer):
    await disable_test_user_transactions()
    balance = await user_credit.get_credits(DEFAULT_USER_ID)
    assert balance == REFILL_VALUE
