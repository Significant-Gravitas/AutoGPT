from datetime import datetime, timezone
from typing import Literal

from prisma.enums import CreditTransactionType
from pydantic import Json, TypeAdapter, field_validator

from backend.data.credit_history_cursor import (
    CreditHistoryCursor,
    as_utc,
    cursor_scope,
    decode_cursor,
    encode_cursor,
)
from backend.data.credit_history_enrichment import enrich_credit_history
from backend.data.credit_history_queries import credit_history_query
from backend.data.db import query_raw_with_schema
from backend.data.model import (
    CreditHistoryCharge,
    CreditTransactionItem,
    TransactionHistory,
)


async def get_credit_history(
    user_id: str,
    transaction_count_limit: int | None = 100,
    transaction_time_ceiling: datetime | None = None,
    transaction_type: str | None = None,
    cursor: str | None = None,
    organization_id: str | None = None,
    viewer_organization_id: str | None = None,
) -> TransactionHistory:
    """Read wallet history with separately scoped viewer enrichment.

    organization_id selects the authorized org wallet; None selects user_id's
    personal wallet. viewer_organization_id scopes execution/chat visibility,
    falling back to organization_id; it never changes the wallet being read.
    """
    limit = transaction_count_limit if transaction_count_limit is not None else 100
    if not 1 <= limit <= 1000:
        raise ValueError("Transaction count limit must be between 1 and 1000")
    if transaction_type is not None:
        CreditTransactionType(transaction_type)
    scope = cursor_scope(user_id, organization_id, transaction_type)
    position = decode_cursor(cursor, scope) if cursor is not None else None
    snapshot = position.snapshot_at if position else datetime.now(timezone.utc)
    ceiling = as_utc(transaction_time_ceiling) if transaction_time_ceiling else None
    rows = await query_raw_with_schema(
        credit_history_query(organization=organization_id is not None),
        organization_id if organization_id is not None else user_id,
        snapshot,
        transaction_type,
        position.transaction_time if position else None,
        position.group_id if position else None,
        ceiling,
        limit + 1,
        model=_HistoryRow,
    )
    selected = [_to_item(row, user_id) for row in rows[:limit]]
    transactions = await enrich_credit_history(
        selected,
        user_id=user_id,
        organization_id=(
            viewer_organization_id
            if viewer_organization_id is not None
            else organization_id
        ),
    )
    for item in transactions:
        if item.activity_type == "agent_run":
            item.description = item.agent_name or "Agent unavailable"
    has_more = len(rows) > limit
    last = rows[limit - 1] if has_more else None
    return TransactionHistory(
        transactions=transactions,
        snapshot_at=as_utc(snapshot),
        next_transaction_time=as_utc(last.transaction_time) if last else None,
        next_cursor=(
            encode_cursor(
                CreditHistoryCursor(
                    snapshot_at=snapshot,
                    transaction_time=as_utc(last.transaction_time),
                    group_id=last.id,
                    scope=scope,
                )
            )
            if last is not None
            else None
        ),
    )


class _HistoryRow(CreditTransactionItem):
    user_id: str = ""
    usage_has_block: bool = False
    usage_is_daily_reset: bool = False

    @field_validator("charges", mode="before")
    @classmethod
    def parse_charges(cls, value: object) -> list[CreditHistoryCharge]:
        # Prisma's typed raw-query decoder serializes JSON columns to strings.
        return _HISTORY_CHARGES.validate_python(value)


def _to_item(row: _HistoryRow, user_id: str) -> CreditTransactionItem:
    activity_type: Literal["agent_run", "copilot_tools", "block_usage", "other"] = (
        "other"
    )
    description = _TRANSACTION_DESCRIPTIONS[row.transaction_type]
    if row.transaction_type == CreditTransactionType.USAGE:
        if row.usage_execution_id and row.usage_execution_id.startswith(
            "copilot-session-"
        ):
            activity_type, description = "copilot_tools", "Autopilot tool use"
        elif row.usage_execution_id:
            activity_type, description = "agent_run", "Agent run"
        elif row.usage_has_block:
            activity_type, description = "block_usage", "Block usage"
        else:
            description = (
                "Daily limit reset" if row.usage_is_daily_reset else "Credit usage"
            )
    return CreditTransactionItem(
        **row.model_dump(
            exclude={
                "user_id",
                "activity_type",
                "description",
                "charges_truncated",
                "transaction_time",
                "usage_start_time",
                "charges",
                "usage_has_block",
                "usage_is_daily_reset",
            }
        ),
        user_id=user_id,
        activity_type=activity_type,
        description=description,
        transaction_time=as_utc(row.transaction_time),
        usage_start_time=as_utc(row.usage_start_time),
        charges=[
            charge.model_copy(update={"posted_at": as_utc(charge.posted_at)})
            for charge in row.charges
        ],
        charges_truncated=row.charges_total_count > len(row.charges),
    )


_TRANSACTION_DESCRIPTIONS = {
    CreditTransactionType.TOP_UP: "Credits added",
    CreditTransactionType.USAGE: "Usage",
    CreditTransactionType.GRANT: "Credits granted",
    CreditTransactionType.REFUND: "Top-up refunded",
    CreditTransactionType.CARD_CHECK: "Card verification",
    CreditTransactionType.SUBSCRIPTION: "Subscription payment",
}

_HISTORY_CHARGES = TypeAdapter(
    list[CreditHistoryCharge] | Json[list[CreditHistoryCharge]]
)
