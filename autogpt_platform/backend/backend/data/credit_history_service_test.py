from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from prisma.enums import CreditTransactionType

from backend.data.credit_history import _HistoryRow, _to_item, get_credit_history
from backend.data.credit_history_cursor import cursor_scope, decode_cursor
from backend.data.model import CreditHistoryCharge, CreditTransactionItem


def history_row(key: str, execution_id: str | None = "run-1") -> _HistoryRow:
    return _HistoryRow(
        id=key,
        transaction_key=key,
        amount=-50,
        transaction_time=datetime(2026, 9, 4, tzinfo=timezone.utc),
        usage_start_time=datetime(2026, 9, 3, tzinfo=timezone.utc),
        usage_execution_id=execution_id,
        usage_graph_id="graph-1",
        usage_has_block=True,
        usage_charge_amount=-100,
        usage_adjustment_amount=50,
        charges_total_count=2,
        charges=[
            CreditHistoryCharge(
                id="charge",
                amount=-100,
                charge_type="usage",
                posted_at=datetime(2026, 9, 3, tzinfo=timezone.utc),
            ),
            CreditHistoryCharge(
                id="adjustment",
                amount=50,
                charge_type="adjustment",
                posted_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
            ),
        ],
    )


@pytest.mark.asyncio
async def test_page_cursor_preserves_snapshot_and_exact_position(monkeypatch):
    query = AsyncMock(
        return_value=[history_row("execution:run-2"), history_row("execution:run-1")]
    )
    enrichment = AsyncMock(side_effect=lambda items, **kwargs: items)
    monkeypatch.setattr("backend.data.credit_history.query_raw_with_schema", query)
    monkeypatch.setattr("backend.data.credit_history.enrich_credit_history", enrichment)
    first = await get_credit_history("user", transaction_count_limit=1)
    assert len(first.transactions) == 1
    assert first.next_cursor
    position = decode_cursor(first.next_cursor, cursor_scope("user", None, None))
    assert position.group_id == "execution:run-2"
    assert first.snapshot_at == position.snapshot_at
    query.return_value = [history_row("execution:run-1")]
    second = await get_credit_history(
        "user", transaction_count_limit=1, cursor=first.next_cursor
    )
    args = query.call_args.args
    assert args[1] == "user"
    assert args[2] == position.snapshot_at
    assert args[4] == position.transaction_time
    assert args[5] == position.group_id
    assert args[7] == 2
    assert second.next_cursor is None
    assert second.snapshot_at == first.snapshot_at
    assert enrichment.call_args.kwargs == {"user_id": "user", "organization_id": None}


@pytest.mark.asyncio
async def test_org_uses_same_query_contract_and_scope(monkeypatch):
    query = AsyncMock(return_value=[history_row("execution:run-1")])
    monkeypatch.setattr("backend.data.credit_history.query_raw_with_schema", query)
    monkeypatch.setattr(
        "backend.data.credit_history.enrich_credit_history",
        AsyncMock(side_effect=lambda items, **kwargs: items),
    )
    ceiling = datetime(2026, 9, 5, 12, tzinfo=timezone(timedelta(hours=2)))
    result = await get_credit_history(
        "viewer",
        organization_id="org-1",
        transaction_type="USAGE",
        transaction_time_ceiling=ceiling,
    )
    args = query.call_args.args
    assert '"OrgCreditTransaction"' in args[0]
    assert '"orgId" = $1' in args[0]
    assert args[1] == "org-1"
    assert args[3] == "USAGE"
    assert args[6] == datetime(2026, 9, 5, 10, tzinfo=timezone.utc)
    assert result.transactions[0].user_id == "viewer"


@pytest.mark.asyncio
async def test_viewer_context_does_not_switch_the_wallet(monkeypatch):
    query = AsyncMock(
        return_value=[history_row("execution:run-2"), history_row("execution:run-1")]
    )
    enrichment = AsyncMock(side_effect=lambda items, **kwargs: items)
    monkeypatch.setattr("backend.data.credit_history.query_raw_with_schema", query)
    monkeypatch.setattr("backend.data.credit_history.enrich_credit_history", enrichment)
    page = await get_credit_history(
        "user", transaction_count_limit=1, viewer_organization_id="current-context"
    )
    args = query.call_args.args
    assert '"CreditTransaction"' in args[0]
    assert '"userId" = $1' in args[0]
    assert args[1] == "user"
    enrichment.assert_awaited_once_with(
        page.transactions, user_id="user", organization_id="current-context"
    )
    assert page.next_cursor
    decode_cursor(page.next_cursor, cursor_scope("user", None, None))


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, 1001])
async def test_invalid_page_limit_is_rejected_before_query(limit):
    with pytest.raises(ValueError, match="Transaction count limit"):
        await get_credit_history("user", transaction_count_limit=limit)


@pytest.mark.parametrize(
    "execution_id,activity,description",
    [
        ("run-1", "agent_run", "Agent run"),
        ("copilot-session-chat", "copilot_tools", "Autopilot tool use"),
        (None, "block_usage", "Block usage"),
    ],
)
def test_usage_types_are_not_mislabelled(execution_id, activity, description):
    item = _to_item(history_row("id", execution_id), "user")
    assert item.activity_type == activity
    assert item.description == description
    assert (
        item.amount
        == item.usage_charge_amount
        + item.usage_fee_amount
        + item.usage_adjustment_amount
    )
    assert not item.charges_truncated


def test_top_up_refund_is_negative_wallet_movement():
    row = history_row("payment-id", None).model_copy(
        update={"transaction_type": CreditTransactionType.REFUND}
    )
    item = _to_item(row, "user")
    assert item.activity_type == "other"
    assert item.description == "Top-up refunded"
    assert item.transaction_key == "payment-id"
    assert item.amount == -50


def test_added_model_fields_preserve_legacy_construction():
    item = CreditTransactionItem(
        user_id="user", transaction_key="payment-id", amount=100
    )
    assert item.charges == []
    assert item.related_executions == []
    assert item.library_agent_id is None
    assert not item.execution_available


@pytest.mark.parametrize(
    "is_reset,expected",
    [(True, "Daily limit reset"), (False, "Credit usage")],
)
def test_reason_only_usage_does_not_invent_a_block(is_reset, expected):
    row = history_row("transaction:key", None).model_copy(
        update={"usage_has_block": False, "usage_is_daily_reset": is_reset}
    )
    item = _to_item(row, "user")
    assert item.activity_type == "other"
    assert item.description == expected
    assert "usage_has_block" not in item.model_dump()
    assert "usage_is_daily_reset" not in item.model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,expected",
    [("Research digest", "Research digest"), (None, "Agent unavailable")],
)
async def test_legacy_description_uses_safe_enriched_name(monkeypatch, name, expected):
    monkeypatch.setattr(
        "backend.data.credit_history.query_raw_with_schema",
        AsyncMock(return_value=[history_row("execution:run")]),
    )

    async def enrich(items, **kwargs):
        return [item.model_copy(update={"agent_name": name}) for item in items]

    monkeypatch.setattr("backend.data.credit_history.enrich_credit_history", enrich)
    result = await get_credit_history("user")
    assert result.transactions[0].description == expected


def test_raw_query_dates_are_unambiguously_utc():
    row = history_row("execution:run").model_copy(
        update={"transaction_time": datetime(2026, 9, 5, 10)}
    )
    row.charges[0].posted_at = datetime(2026, 9, 5, 9)
    item = _to_item(row, "user")
    assert item.transaction_time == datetime(2026, 9, 5, 10, tzinfo=timezone.utc)
    assert item.charges[0].posted_at == datetime(2026, 9, 5, 9, tzinfo=timezone.utc)
