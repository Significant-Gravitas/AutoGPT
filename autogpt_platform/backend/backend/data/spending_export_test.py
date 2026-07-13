"""Unit tests for the admin CSV export functions on credit + platform_cost.

Mocks prisma + raw-SQL helpers so these run fast without the docker postgres.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import CreditTransactionType

from backend.data.credit import (
    CREDIT_EXPORT_MAX_DAYS,
    CREDIT_EXPORT_MAX_ROWS,
    admin_export_user_history,
)
from backend.data.platform_cost import (
    COPILOT_USAGE_EXPORT_MAX_DAYS,
    get_copilot_weekly_usage_for_export,
)


def _make_tx(
    *,
    user_id: str = "u1",
    amount: int = 100,
    running_balance: int = 1000,
    metadata: dict | None = None,
    user_email: str | None = "u1@example.com",
    tx_type: CreditTransactionType = CreditTransactionType.GRANT,
    created_at: datetime | None = None,
) -> MagicMock:
    tx = MagicMock()
    tx.transactionKey = "tx-key"
    tx.createdAt = created_at or datetime(2026, 4, 1, tzinfo=timezone.utc)
    tx.type = tx_type
    tx.amount = amount
    tx.runningBalance = running_balance
    tx.userId = user_id
    tx.metadata = metadata or {}
    user = MagicMock()
    user.email = user_email
    tx.User = user if user_email is not None else None
    return tx


class TestAdminExportUserHistory:
    @pytest.mark.asyncio
    async def test_rejects_inverted_window(self):
        with pytest.raises(ValueError, match="end must be >= start"):
            await admin_export_user_history(
                start=datetime(2026, 4, 30, tzinfo=timezone.utc),
                end=datetime(2026, 4, 1, tzinfo=timezone.utc),
            )

    @pytest.mark.asyncio
    async def test_rejects_window_exceeding_cap_by_subday_remainder(self):
        # 90 days + 1 second — `.days` would round to 90 and pass; timedelta
        # comparison must still reject this.
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=CREDIT_EXPORT_MAX_DAYS, seconds=1)
        with pytest.raises(ValueError, match="must be <="):
            await admin_export_user_history(start=start, end=end)

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_rows(self):
        with patch("backend.data.credit.CreditTransaction") as ct, patch(
            "backend.data.credit.get_user_email_by_id",
            new_callable=AsyncMock,
        ):
            ct.prisma.return_value.find_many = AsyncMock(return_value=[])
            result = await admin_export_user_history(
                start=datetime(2026, 4, 1, tzinfo=timezone.utc),
                end=datetime(2026, 4, 30, tzinfo=timezone.utc),
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_maps_basic_row(self):
        tx = _make_tx(metadata={"reason": "Initial grant"})
        with patch("backend.data.credit.CreditTransaction") as ct, patch(
            "backend.data.credit.get_user_email_by_id",
            new_callable=AsyncMock,
        ):
            ct.prisma.return_value.find_many = AsyncMock(return_value=[tx])
            result = await admin_export_user_history(
                start=datetime(2026, 4, 1, tzinfo=timezone.utc),
                end=datetime(2026, 4, 30, tzinfo=timezone.utc),
            )
        assert len(result) == 1
        assert result[0].user_id == "u1"
        assert result[0].user_email == "u1@example.com"
        assert result[0].amount == 100
        assert result[0].running_balance == 1000
        assert result[0].reason == "Initial grant"
        assert result[0].admin_email == ""

    @pytest.mark.asyncio
    async def test_unwraps_nested_reason_from_top_up_credits_metadata(self):
        # _top_up_credits writes reason as {"reason": "..."} — the export
        # must flatten this for the CSV column.
        tx = _make_tx(
            tx_type=CreditTransactionType.TOP_UP,
            metadata={"reason": {"reason": "Auto top up"}},
        )
        with patch("backend.data.credit.CreditTransaction") as ct, patch(
            "backend.data.credit.get_user_email_by_id",
            new_callable=AsyncMock,
        ):
            ct.prisma.return_value.find_many = AsyncMock(return_value=[tx])
            result = await admin_export_user_history(
                start=datetime(2026, 4, 1, tzinfo=timezone.utc),
                end=datetime(2026, 4, 30, tzinfo=timezone.utc),
            )
        assert result[0].reason == "Auto top up"

    @pytest.mark.asyncio
    async def test_resolves_admin_email_once_per_unique_admin(self):
        # Two rows from same admin → email lookup hits cache on second.
        tx_a = _make_tx(user_id="u1", metadata={"admin_id": "admin-1"})
        tx_b = _make_tx(user_id="u2", metadata={"admin_id": "admin-1"})
        email_lookup = AsyncMock(return_value="admin1@example.com")
        with patch("backend.data.credit.CreditTransaction") as ct, patch(
            "backend.data.credit.get_user_email_by_id",
            email_lookup,
        ):
            ct.prisma.return_value.find_many = AsyncMock(return_value=[tx_a, tx_b])
            result = await admin_export_user_history(
                start=datetime(2026, 4, 1, tzinfo=timezone.utc),
                end=datetime(2026, 4, 30, tzinfo=timezone.utc),
            )
        assert email_lookup.call_count == 1
        assert all(r.admin_email == "admin1@example.com" for r in result)

    @pytest.mark.asyncio
    async def test_admin_email_empty_when_lookup_misses(self):
        # When get_user_email_by_id returns None, admin_email is "" not "Unknown
        # Admin: <uuid>" so the column doesn't mix email + UUID formats.
        tx = _make_tx(metadata={"admin_id": "deleted-admin"})
        with patch("backend.data.credit.CreditTransaction") as ct, patch(
            "backend.data.credit.get_user_email_by_id",
            AsyncMock(return_value=None),
        ):
            ct.prisma.return_value.find_many = AsyncMock(return_value=[tx])
            result = await admin_export_user_history(
                start=datetime(2026, 4, 1, tzinfo=timezone.utc),
                end=datetime(2026, 4, 30, tzinfo=timezone.utc),
            )
        assert result[0].admin_email == ""

    @pytest.mark.asyncio
    async def test_passes_filters_to_prisma_where_clause(self):
        find_many = AsyncMock(return_value=[])
        with patch("backend.data.credit.CreditTransaction") as ct, patch(
            "backend.data.credit.get_user_email_by_id",
            new_callable=AsyncMock,
        ):
            ct.prisma.return_value.find_many = find_many
            await admin_export_user_history(
                start=datetime(2026, 4, 1, tzinfo=timezone.utc),
                end=datetime(2026, 4, 30, tzinfo=timezone.utc),
                transaction_type=CreditTransactionType.TOP_UP,
                user_id="u1",
            )
        where = find_many.call_args.kwargs["where"]
        assert where["type"] == CreditTransactionType.TOP_UP
        assert where["userId"] == "u1"
        assert "createdAt" in where
        # take must be cap+1 to detect over-cap rowsets without a separate count.
        assert find_many.call_args.kwargs["take"] == CREDIT_EXPORT_MAX_ROWS + 1

    @pytest.mark.asyncio
    async def test_filters_inactive_rows_by_default(self):
        find_many = AsyncMock(return_value=[])
        with patch("backend.data.credit.CreditTransaction") as ct, patch(
            "backend.data.credit.get_user_email_by_id",
            new_callable=AsyncMock,
        ):
            ct.prisma.return_value.find_many = find_many
            await admin_export_user_history(
                start=datetime(2026, 4, 1, tzinfo=timezone.utc),
                end=datetime(2026, 4, 30, tzinfo=timezone.utc),
            )
        assert find_many.call_args.kwargs["where"]["isActive"] is True

    @pytest.mark.asyncio
    async def test_include_inactive_drops_isactive_filter(self):
        find_many = AsyncMock(return_value=[])
        with patch("backend.data.credit.CreditTransaction") as ct, patch(
            "backend.data.credit.get_user_email_by_id",
            new_callable=AsyncMock,
        ):
            ct.prisma.return_value.find_many = find_many
            await admin_export_user_history(
                start=datetime(2026, 4, 1, tzinfo=timezone.utc),
                end=datetime(2026, 4, 30, tzinfo=timezone.utc),
                include_inactive=True,
            )
        assert "isActive" not in find_many.call_args.kwargs["where"]

    @pytest.mark.asyncio
    async def test_rejects_when_fetched_rowset_exceeds_cap(self):
        oversize = [_make_tx() for _ in range(CREDIT_EXPORT_MAX_ROWS + 1)]
        with patch("backend.data.credit.CreditTransaction") as ct, patch(
            "backend.data.credit.get_user_email_by_id",
            new_callable=AsyncMock,
        ):
            ct.prisma.return_value.find_many = AsyncMock(return_value=oversize)
            with pytest.raises(ValueError, match="more than"):
                await admin_export_user_history(
                    start=datetime(2026, 4, 1, tzinfo=timezone.utc),
                    end=datetime(2026, 4, 30, tzinfo=timezone.utc),
                )


class TestGetCopilotWeeklyUsageForExport:
    @pytest.mark.asyncio
    async def test_rejects_inverted_window(self):
        with pytest.raises(ValueError, match="end must be >= start"):
            await get_copilot_weekly_usage_for_export(
                start=datetime(2026, 4, 30, tzinfo=timezone.utc),
                end=datetime(2026, 4, 1, tzinfo=timezone.utc),
            )

    @pytest.mark.asyncio
    async def test_rejects_window_exceeding_cap_by_subday_remainder(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=COPILOT_USAGE_EXPORT_MAX_DAYS, hours=2)
        with pytest.raises(ValueError, match="must be <="):
            await get_copilot_weekly_usage_for_export(start=start, end=end)

    @pytest.mark.asyncio
    async def test_empty_result_when_no_copilot_usage(self):
        with patch(
            "backend.data.platform_cost.query_raw_with_schema",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await get_copilot_weekly_usage_for_export(
                start=datetime(2026, 4, 1, tzinfo=timezone.utc),
                end=datetime(2026, 4, 30, tzinfo=timezone.utc),
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_computes_inclusive_week_end_and_percent_used(self):
        rows = [
            {
                "user_id": "u1",
                "user_email": "u1@example.com",
                "tier": "PRO",
                "week_start": datetime(2026, 3, 30, tzinfo=timezone.utc),
                "cost_microdollars": 5_000_000,
            }
        ]
        with patch(
            "backend.data.platform_cost.query_raw_with_schema",
            new_callable=AsyncMock,
            return_value=rows,
        ), patch(
            "backend.copilot.rate_limit.get_tier_multipliers",
            new_callable=AsyncMock,
            return_value={"PRO": 5.0, "BETA": 1.0, "NO_TIER": 0.0},
        ):
            result = await get_copilot_weekly_usage_for_export(
                start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                end=datetime(2026, 4, 30, tzinfo=timezone.utc),
            )
        assert len(result) == 1
        row = result[0]
        assert row.user_id == "u1"
        assert row.tier == "PRO"
        # week_end is inclusive end-of-Sunday, NOT next Monday.
        assert row.week_end == datetime(
            2026, 4, 5, 23, 59, 59, 999999, tzinfo=timezone.utc
        )
        assert row.copilot_cost_microdollars == 5_000_000

    @pytest.mark.asyncio
    async def test_no_tier_user_gets_zero_percent(self):
        rows = [
            {
                "user_id": "u1",
                "user_email": "u1@example.com",
                "tier": "NO_TIER",
                "week_start": datetime(2026, 3, 30, tzinfo=timezone.utc),
                "cost_microdollars": 1_000_000,
            }
        ]
        with patch(
            "backend.data.platform_cost.query_raw_with_schema",
            new_callable=AsyncMock,
            return_value=rows,
        ), patch(
            "backend.copilot.rate_limit.get_tier_multipliers",
            new_callable=AsyncMock,
            return_value={"PRO": 5.0, "BETA": 1.0, "NO_TIER": 0.0},
        ):
            result = await get_copilot_weekly_usage_for_export(
                start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                end=datetime(2026, 4, 30, tzinfo=timezone.utc),
            )
        assert result[0].weekly_limit_microdollars == 0
        assert result[0].percent_used == 0.0

    @pytest.mark.asyncio
    async def test_unknown_tier_falls_back_to_default(self):
        rows = [
            {
                "user_id": "u1",
                "user_email": "u1@example.com",
                "tier": "MARS",  # not a real SubscriptionTier value
                "week_start": datetime(2026, 3, 30, tzinfo=timezone.utc),
                "cost_microdollars": 100_000,
            }
        ]
        with patch(
            "backend.data.platform_cost.query_raw_with_schema",
            new_callable=AsyncMock,
            return_value=rows,
        ), patch(
            "backend.copilot.rate_limit.get_tier_multipliers",
            new_callable=AsyncMock,
            return_value={"PRO": 5.0, "BETA": 1.0, "NO_TIER": 0.0},
        ):
            result = await get_copilot_weekly_usage_for_export(
                start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                end=datetime(2026, 4, 30, tzinfo=timezone.utc),
            )
        # Falls back to DEFAULT_TIER (NO_TIER)
        assert result[0].tier == "NO_TIER"

    @pytest.mark.asyncio
    async def test_naive_week_start_gets_utc_tzinfo(self):
        rows = [
            {
                "user_id": "u1",
                "user_email": "u1@example.com",
                "tier": "PRO",
                "week_start": datetime(2026, 3, 30),  # naive
                "cost_microdollars": 1,
            }
        ]
        with patch(
            "backend.data.platform_cost.query_raw_with_schema",
            new_callable=AsyncMock,
            return_value=rows,
        ), patch(
            "backend.copilot.rate_limit.get_tier_multipliers",
            new_callable=AsyncMock,
            return_value={"PRO": 5.0},
        ):
            result = await get_copilot_weekly_usage_for_export(
                start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                end=datetime(2026, 4, 30, tzinfo=timezone.utc),
            )
        assert result[0].week_start.tzinfo == timezone.utc
