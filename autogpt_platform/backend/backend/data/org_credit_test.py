"""Tests for org-level credit operations.

Tests the atomic spend/top-up logic, edge cases, and error paths.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.org_credit import (
    assign_seat,
    get_org_credits,
    get_org_transaction_history,
    get_seat_info,
    spend_org_credits,
    top_up_org_credits,
    unassign_seat,
)
from backend.util.exceptions import InsufficientBalanceError


@pytest.fixture(autouse=True)
def mock_prisma(mocker):
    mock = MagicMock()
    mock.orgbalance.find_unique = AsyncMock(return_value=MagicMock(balance=1000))
    mock.query_raw = AsyncMock(return_value=[{"balance": 1000}])  # RETURNING data
    mock.orgcredittransaction.create = AsyncMock()
    mock.orgcredittransaction.find_many = AsyncMock(return_value=[])
    mock.organizationseatassignment.find_many = AsyncMock(return_value=[])
    mock.organizationseatassignment.upsert = AsyncMock(
        return_value=MagicMock(userId="u1", seatType="FREE", status="ACTIVE")
    )
    mock.organizationseatassignment.update = AsyncMock()
    mocker.patch("backend.data.org_credit.prisma", mock)
    return mock


class TestGetOrgCredits:
    @pytest.mark.asyncio
    async def test_returns_balance(self, mock_prisma):
        result = await get_org_credits("org-1")
        assert result == 1000

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_balance_row(self, mock_prisma):
        mock_prisma.orgbalance.find_unique = AsyncMock(return_value=None)
        result = await get_org_credits("org-missing")
        assert result == 0


class TestSpendOrgCredits:
    @pytest.mark.asyncio
    async def test_spend_success_returns_remaining(self, mock_prisma):
        # query_raw returns RETURNING data with new balance
        mock_prisma.query_raw = AsyncMock(return_value=[{"balance": 900}])

        result = await spend_org_credits("org-1", "user-1", 100)
        assert result == 900
        mock_prisma.orgcredittransaction.create.assert_called_once()
        # Verify transaction data is correct
        tx_data = mock_prisma.orgcredittransaction.create.call_args[1]["data"]
        assert tx_data["orgId"] == "org-1"
        assert tx_data["initiatedByUserId"] == "user-1"
        assert tx_data["amount"] == -100
        assert tx_data["runningBalance"] == 900

    @pytest.mark.asyncio
    async def test_spend_insufficient_balance_raises(self, mock_prisma):
        # query_raw returns empty list = no row matched (insufficient balance)
        mock_prisma.query_raw = AsyncMock(return_value=[])
        mock_prisma.orgbalance.find_unique = AsyncMock(
            return_value=MagicMock(balance=50)
        )

        with pytest.raises(InsufficientBalanceError) as exc_info:
            await spend_org_credits("org-1", "user-1", 100)

        assert exc_info.value.balance == 50
        assert exc_info.value.amount == 100
        # Transaction should NOT be created on failure
        mock_prisma.orgcredittransaction.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_spend_zero_amount_raises(self):
        with pytest.raises(ValueError, match="positive"):
            await spend_org_credits("org-1", "user-1", 0)

    @pytest.mark.asyncio
    async def test_spend_negative_amount_raises(self):
        with pytest.raises(ValueError, match="positive"):
            await spend_org_credits("org-1", "user-1", -5)

    @pytest.mark.asyncio
    async def test_spend_records_workspace_attribution(self, mock_prisma):
        mock_prisma.query_raw = AsyncMock(return_value=[{"balance": 800}])

        await spend_org_credits(
            "org-1", "user-1", 200, team_id="ws-1", metadata={"block": "llm"}
        )

        tx_data = mock_prisma.orgcredittransaction.create.call_args[1]["data"]
        assert tx_data["teamId"] == "ws-1"
        assert tx_data["amount"] == -200


class TestTopUpOrgCredits:
    @pytest.mark.asyncio
    async def test_top_up_success(self, mock_prisma):
        # query_raw returns RETURNING data with new balance
        mock_prisma.query_raw = AsyncMock(return_value=[{"balance": 1500}])

        result = await top_up_org_credits("org-1", 500, user_id="user-1")
        assert result == 1500
        mock_prisma.query_raw.assert_called_once()  # Atomic upsert
        # Verify transaction data
        mock_prisma.orgcredittransaction.create.assert_called_once()
        tx_data = mock_prisma.orgcredittransaction.create.call_args[1]["data"]
        assert tx_data["orgId"] == "org-1"
        assert tx_data["amount"] == 500
        assert tx_data["initiatedByUserId"] == "user-1"
        mock_prisma.orgcredittransaction.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_top_up_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            await top_up_org_credits("org-1", 0)

    @pytest.mark.asyncio
    async def test_top_up_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            await top_up_org_credits("org-1", -10)

    @pytest.mark.asyncio
    async def test_top_up_no_user_id_omits_from_transaction(self, mock_prisma):
        # query_raw returns RETURNING data with new balance
        mock_prisma.query_raw = AsyncMock(return_value=[{"balance": 500}])

        await top_up_org_credits("org-1", 500)

        tx_data = mock_prisma.orgcredittransaction.create.call_args[1]["data"]
        assert "initiatedByUserId" not in tx_data


class TestGetOrgTransactionHistory:
    @pytest.mark.asyncio
    async def test_returns_transactions(self, mock_prisma):
        mock_tx = MagicMock(
            transactionKey="tx-1",
            createdAt="2026-01-01",
            amount=-100,
            type="USAGE",
            runningBalance=900,
            initiatedByUserId="user-1",
            teamId="ws-1",
            metadata=None,
        )
        mock_prisma.orgcredittransaction.find_many = AsyncMock(return_value=[mock_tx])

        result = await get_org_transaction_history("org-1", limit=10)
        assert len(result) == 1
        assert result[0]["amount"] == -100
        assert result[0]["teamId"] == "ws-1"


class TestSeatManagement:
    @pytest.mark.asyncio
    async def test_get_seat_info(self, mock_prisma):
        mock_prisma.organizationseatassignment.find_many = AsyncMock(
            return_value=[
                MagicMock(
                    userId="u1", seatType="PAID", status="ACTIVE", createdAt="now"
                ),
                MagicMock(
                    userId="u2", seatType="FREE", status="INACTIVE", createdAt="now"
                ),
            ]
        )

        result = await get_seat_info("org-1")
        assert result["total"] == 2
        assert result["active"] == 1
        assert result["inactive"] == 1
        # Verify the query filtered by org
        call_kwargs = mock_prisma.organizationseatassignment.find_many.call_args[1]
        assert call_kwargs["where"]["organizationId"] == "org-1"

    @pytest.mark.asyncio
    async def test_assign_seat(self, mock_prisma):
        mock_prisma.organizationseatassignment.upsert = AsyncMock(
            return_value=MagicMock(userId="user-1", seatType="PAID", status="ACTIVE")
        )
        result = await assign_seat("org-1", "user-1", seat_type="PAID")
        assert result["seatType"] == "PAID"
        mock_prisma.organizationseatassignment.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_unassign_seat(self, mock_prisma):
        await unassign_seat("org-1", "user-1")
        mock_prisma.organizationseatassignment.update.assert_called_once()
        call_kwargs = mock_prisma.organizationseatassignment.update.call_args[1]
        # Verify correct record targeted
        where = call_kwargs["where"]["organizationId_userId"]
        assert where["organizationId"] == "org-1"
        assert where["userId"] == "user-1"
        # Verify status set to INACTIVE
        assert call_kwargs["data"]["status"] == "INACTIVE"
