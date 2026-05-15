import pytest
from prisma.enums import CreditTransactionType
from prisma.models import CreditTransaction, UserBalance

from backend.blocks import get_block
from backend.blocks.llm import AITextGeneratorBlock
from backend.data.credit import UsageTransactionMetadata, UserCredit
from backend.data.execution import ExecutionContext, NodeExecutionEntry
from backend.data.user import DEFAULT_USER_ID
from backend.executor.utils import block_usage_cost
from backend.integrations.credentials_store import openai_credentials
from backend.util.test import SpinTestServer

user_credit = UserCredit()


async def disable_test_user_transactions():
    await CreditTransaction.prisma().delete_many(where={"userId": DEFAULT_USER_ID})
    await UserBalance.prisma().upsert(
        where={"userId": DEFAULT_USER_ID},
        data={
            "create": {"userId": DEFAULT_USER_ID, "balance": 0},
            "update": {"balance": 0},
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
