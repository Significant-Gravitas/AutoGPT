from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.enums import CreditTransactionType
from prisma.models import CreditTransaction, Organization, OrgCreditTransaction, User

from backend.blocks.jina.search import SearchTheWebBlock
from backend.data import credit_metadata
from backend.data.credit import UsageTransactionMetadata
from backend.data.credit_history import get_credit_history
from backend.data.credit_history.queries import credit_history_query
from backend.data.db import get_database_schema, prisma
from backend.data.execution import ExecutionContext, NodeExecutionEntry
from backend.data.model import NodeExecutionStats
from backend.executor.billing import charge_reconciled_usage, charge_usage
from backend.util.json import SafeJson


@pytest_asyncio.fixture(loop_scope="session")
async def history_wallet(server, monkeypatch):
    user_id = str(uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"history-{user_id}@example.com"}
    )
    monkeypatch.setattr(
        "backend.data.credit_history.service.enrich_credit_history",
        AsyncMock(side_effect=lambda items, **kwargs: items),
    )
    yield user_id
    await CreditTransaction.prisma().delete_many(where={"userId": user_id})
    await User.prisma().delete(where={"id": user_id})


async def add_raw_charge(
    user_id: str,
    key: str,
    amount: int,
    time: datetime,
    metadata: dict,
):
    return await CreditTransaction.prisma().create(
        data={
            "userId": user_id,
            "transactionKey": key,
            "createdAt": time,
            "amount": amount,
            "type": CreditTransactionType.USAGE,
            "metadata": SafeJson(metadata),
        }
    )


async def add_charge(
    user_id: str,
    key: str,
    amount: int,
    time: datetime,
    execution_id: str | None = "run-1",
    adjustment: bool = False,
    fee: bool = False,
):
    metadata = {
        "graph_exec_id": execution_id,
        "graph_id": "graph-1",
        "node_exec_id": None if fee else "node-1",
        "block": "AITextGeneratorBlock",
        "input": {"secret": "never-return-this"},
    }
    if adjustment:
        metadata["input"]["reconciled_delta"] = -amount
    if fee:
        metadata["input"]["charge"] = "Execution Cost"
    return await add_raw_charge(user_id, key, amount, time, metadata)


@pytest.mark.asyncio(loop_scope="session")
async def test_writer_metadata_round_trips_through_history_classifier(
    history_wallet, monkeypatch
):
    execution_id = str(uuid4())
    node_exec = NodeExecutionEntry(
        user_id=history_wallet,
        graph_exec_id=execution_id,
        graph_id=str(uuid4()),
        graph_version=1,
        node_exec_id=str(uuid4()),
        node_id=str(uuid4()),
        block_id=SearchTheWebBlock().id,
        inputs={},
        execution_context=ExecutionContext(),
    )
    writes: list[tuple[int, UsageTransactionMetadata]] = []
    sync_db = MagicMock()
    sync_db.spend_credits.side_effect = lambda *, cost, metadata, **_: writes.append(
        (-cost, metadata)
    )
    async_db = MagicMock()
    async_db.spend_credits = AsyncMock(
        side_effect=lambda *, cost, metadata, **_: writes.append((-cost, metadata)) or 0
    )
    block = SearchTheWebBlock()
    monkeypatch.setattr(
        "backend.executor.billing.resolve_block_cost", lambda _: (block, 0, {})
    )
    monkeypatch.setattr(
        "backend.executor.billing._block_has_paid_cost_entry", lambda *_: False
    )
    monkeypatch.setattr(
        "backend.executor.billing.execution_usage_cost", lambda _: (2, 10)
    )
    monkeypatch.setattr("backend.executor.billing.get_db_client", lambda: sync_db)

    charge_usage(node_exec, execution_count=10)
    with (
        patch("backend.executor.billing.get_block", return_value=block),
        patch(
            "backend.executor.billing.block_usage_cost", side_effect=[(0, {}), (5, {})]
        ),
        patch(
            "backend.executor.billing.get_database_manager_async_client",
            return_value=async_db,
        ),
        patch("backend.executor.billing.handle_low_balance"),
    ):
        assert await charge_reconciled_usage(node_exec, NodeExecutionStats()) == (5, 0)

    time = datetime.now(timezone.utc) - timedelta(minutes=5)
    for index, (amount, metadata) in enumerate(writes):
        await add_raw_charge(
            history_wallet,
            f"writer-{index}",
            amount,
            time + timedelta(seconds=index),
            metadata.model_dump(exclude_none=True),
        )

    row = next(
        item
        for item in (await get_credit_history(history_wallet)).transactions
        if item.usage_execution_id == execution_id
    )
    assert row.amount == -7
    assert row.usage_fee_amount == -2
    assert row.usage_adjustment_amount == -5


@pytest.mark.asyncio(loop_scope="session")
async def test_literal_v1_historical_markers_remain_recognized(
    history_wallet, monkeypatch
):
    monkeypatch.setattr(
        credit_metadata,
        "CURRENT_CREDIT_MARKERS",
        credit_metadata.CreditMetadataMarkers(
            reconciliation_delta_input_key="reconciled_delta_v2",
            execution_fee_input_key="charge_v2",
            execution_fee_input_value="Execution Cost v2",
            daily_reset_reason="CoPilot daily rate limit reset v2",
            copilot_session_prefix="copilot-session-v2-",
        ),
    )
    time = datetime.now(timezone.utc) - timedelta(minutes=5)
    base = {"graph_exec_id": "run-v1", "graph_id": "graph-v1"}
    await add_raw_charge(
        history_wallet,
        "v1-usage",
        -100,
        time,
        {**base, "node_exec_id": "node-v1", "input": {}},
    )
    await add_raw_charge(
        history_wallet,
        "v1-adjustment",
        25,
        time + timedelta(seconds=1),
        {**base, "input": {"reconciled_delta": -25}},
    )
    await add_raw_charge(
        history_wallet,
        "v1-fee",
        -2,
        time + timedelta(seconds=2),
        {**base, "input": {"charge": "Execution Cost"}},
    )
    await add_raw_charge(
        history_wallet,
        "v1-reset",
        -10,
        time + timedelta(seconds=3),
        {"reason": "CoPilot daily rate limit reset", "input": {}},
    )
    await add_raw_charge(
        history_wallet,
        "v1-copilot",
        -5,
        time + timedelta(seconds=4),
        {"graph_exec_id": "copilot-session-v1", "input": {}},
    )

    rows = (await get_credit_history(history_wallet)).transactions
    run = next(row for row in rows if row.usage_execution_id == "run-v1")
    reset = next(row for row in rows if row.transaction_key == "v1-reset")
    copilot = next(
        row for row in rows if row.usage_execution_id == "copilot-session-v1"
    )

    query = credit_history_query(organization=False)
    assert "reconciled_delta_v2" in query
    assert "Execution Cost v2" in query
    assert "CoPilot daily rate limit reset v2" in query

    assert run.amount == -77
    assert run.usage_charge_amount == -100
    assert run.usage_adjustment_amount == 25
    assert run.usage_fee_amount == -2
    assert reset.description == "Daily limit reset"
    assert copilot.activity_type == "copilot_tools"


@pytest.mark.asyncio(loop_scope="session")
async def test_complete_execution_before_pagination(history_wallet):
    now = datetime.now(timezone.utc) - timedelta(minutes=5)
    await add_charge(history_wallet, "estimate", -100, now)
    await add_charge(
        history_wallet, "other-run", -20, now + timedelta(seconds=1), "run-2"
    )
    await add_charge(
        history_wallet, "refund", 40, now + timedelta(seconds=2), adjustment=True
    )
    await add_charge(history_wallet, "fee", -1, now + timedelta(seconds=3), fee=True)

    page = await get_credit_history(history_wallet, transaction_count_limit=1)

    assert len(page.transactions) == 1
    row = page.transactions[0]
    assert row.amount == -61
    assert row.usage_charge_amount == -100
    assert row.usage_fee_amount == -1
    assert row.usage_adjustment_amount == 40
    assert row.usage_node_count == 1
    assert row.charges_total_count == 3
    assert "never-return-this" not in row.model_dump_json()
    assert page.next_cursor
    next_page = await get_credit_history(
        history_wallet, transaction_count_limit=1, cursor=page.next_cursor
    )
    assert [row.usage_execution_id for row in next_page.transactions] == ["run-2"]
    assert next_page.next_cursor is None


@pytest.mark.asyncio(loop_scope="session")
async def test_cursor_ties_and_snapshot_ignore_later_adjustments(history_wallet):
    time = datetime.now(timezone.utc) - timedelta(minutes=5)
    for execution_id in ["run-a", "run-b", "run-c"]:
        await add_charge(history_wallet, execution_id, -10, time, execution_id)
    first = await get_credit_history(history_wallet, transaction_count_limit=1)
    assert first.next_cursor
    await add_charge(
        history_wallet, "later", 5, datetime.now(timezone.utc), "run-a", adjustment=True
    )
    second = await get_credit_history(
        history_wallet, transaction_count_limit=1, cursor=first.next_cursor
    )
    assert second.next_cursor
    third = await get_credit_history(
        history_wallet, transaction_count_limit=1, cursor=second.next_cursor
    )
    all_rows = first.transactions + second.transactions + third.transactions
    assert {row.usage_execution_id for row in all_rows} == {"run-a", "run-b", "run-c"}
    assert all(row.amount == -10 for row in all_rows)
    assert len({row.id for row in all_rows}) == 3


@pytest.mark.asyncio(loop_scope="session")
async def test_bounded_details_keep_complete_total(history_wallet):
    time = datetime.now(timezone.utc) - timedelta(minutes=5)
    for index in range(105):
        await add_charge(history_wallet, f"charge-{index}", -1, time)
    row = (await get_credit_history(history_wallet)).transactions[0]
    assert row.amount == -105
    assert row.charges_total_count == 105
    assert row.charges_truncated
    assert len(row.charges) == 100


@pytest.mark.asyncio(loop_scope="session")
async def test_copilot_and_orphan_usage_are_distinct(history_wallet):
    time = datetime.now(timezone.utc) - timedelta(minutes=5)
    await add_charge(history_wallet, "tool", -10, time, "copilot-session-chat-1")
    await add_charge(history_wallet, "orphan-1", -4, time, None)
    await add_charge(history_wallet, "orphan-2", -5, time, None)
    page = await get_credit_history(history_wallet)
    assert len(page.transactions) == 3
    assert sorted(row.activity_type for row in page.transactions) == [
        "block_usage",
        "block_usage",
        "copilot_tools",
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_invalid_or_cross_wallet_cursor_is_rejected(history_wallet):
    time = datetime.now(timezone.utc) - timedelta(minutes=5)
    for execution_id in ["run-a", "run-b"]:
        await add_charge(history_wallet, execution_id, -1, time, execution_id)
    page = await get_credit_history(history_wallet, transaction_count_limit=1)
    with pytest.raises(ValueError, match="cursor"):
        await get_credit_history("another-wallet", cursor=page.next_cursor)
    with pytest.raises(ValueError, match="cursor"):
        await get_credit_history(history_wallet, cursor="not-a-cursor")


@pytest.mark.asyncio(loop_scope="session")
async def test_timestamp_comparisons_ignore_database_session_timezone(history_wallet):
    time = datetime(2026, 9, 5, 10, tzinfo=timezone.utc)
    await add_charge(history_wallet, "timezone-charge", -10, time)
    schema = get_database_schema()
    query = credit_history_query(organization=False).format(
        schema_prefix=f'"{schema}".' if schema != "public" else "",
        schema=schema,
    )
    before = time - timedelta(hours=1)
    after = time + timedelta(hours=1)
    async with prisma.tx() as tx:
        await tx.execute_raw("SET LOCAL TIME ZONE 'Europe/Paris'")
        assert (
            await tx.query_raw(query, history_wallet, before, None, None, None, None, 2)
            == []
        )
        rows = await tx.query_raw(
            query, history_wallet, after, None, None, None, None, 2
        )
        assert len(rows) == 1
        assert (
            await tx.query_raw(
                query, history_wallet, after, None, before, "execution:run-1", None, 2
            )
            == []
        )
        assert (
            await tx.query_raw(
                query, history_wallet, after, None, None, None, before, 2
            )
            == []
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_org_aggregation_preserves_amounts_and_wallet_boundary(history_wallet):
    org = await Organization.prisma().create(
        data={"name": "History test", "slug": f"history-{uuid4()}"}
    )
    time = (datetime.now(timezone.utc) - timedelta(minutes=5)).replace(microsecond=0)
    try:
        await add_charge(history_wallet, "personal", -999, time)
        for key, amount, is_adjustment in [
            ("estimate", -100, False),
            ("extra", -15, True),
            ("refund", 40, True),
        ]:
            await OrgCreditTransaction.prisma().create(
                data={
                    "orgId": org.id,
                    "transactionKey": key,
                    "createdAt": time,
                    "amount": amount,
                    "type": CreditTransactionType.USAGE,
                    "metadata": SafeJson(
                        {
                            "graph_exec_id": "run-1",
                            "graph_id": "graph-1",
                            "input": (
                                {"reconciled_delta": -amount} if is_adjustment else {}
                            ),
                        }
                    ),
                }
            )
        result = await get_credit_history(history_wallet, organization_id=org.id)
        assert len(result.transactions) == 1
        row = result.transactions[0]
        assert row.amount == -75
        assert row.usage_charge_amount == -100
        assert row.usage_adjustment_amount == 25
        assert row.charges_total_count == 3
        assert row.transaction_time == time
    finally:
        await OrgCreditTransaction.prisma().delete_many(where={"orgId": org.id})
        await Organization.prisma().delete(where={"id": org.id})
