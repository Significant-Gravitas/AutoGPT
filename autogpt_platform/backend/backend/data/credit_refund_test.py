"""
Tests for credit system refund and dispute operations.

These tests ensure that refund operations (deduct_credits, handle_dispute)
are atomic and maintain data consistency.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import stripe
from prisma.enums import CreditTransactionType
from prisma.models import CreditRefundRequest, CreditTransaction, User, UserBalance

from backend.data.credit import UserCredit
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer

credit_system = UserCredit()

# Test user ID for refund tests
REFUND_TEST_USER_ID = "refund-test-user"


async def setup_test_user_with_topup():
    """Create a test user with initial balance and a top-up transaction."""
    # Clean up any existing data
    await CreditRefundRequest.prisma().delete_many(
        where={"userId": REFUND_TEST_USER_ID}
    )
    await CreditTransaction.prisma().delete_many(where={"userId": REFUND_TEST_USER_ID})
    await UserBalance.prisma().delete_many(where={"userId": REFUND_TEST_USER_ID})
    await User.prisma().delete_many(where={"id": REFUND_TEST_USER_ID})

    # Create user
    await User.prisma().create(
        data={
            "id": REFUND_TEST_USER_ID,
            "email": f"{REFUND_TEST_USER_ID}@example.com",
            "name": "Refund Test User",
        }
    )

    # Create user balance
    await UserBalance.prisma().create(
        data={
            "userId": REFUND_TEST_USER_ID,
            "balance": 1000,  # $10
        }
    )

    # Create a top-up transaction that can be refunded
    topup_tx = await CreditTransaction.prisma().create(
        data={
            "userId": REFUND_TEST_USER_ID,
            "amount": 1000,
            "type": CreditTransactionType.TOP_UP,
            "transactionKey": "pi_test_12345",
            "runningBalance": 1000,
            "isActive": True,
            "metadata": SafeJson({"stripe_payment_intent": "pi_test_12345"}),
        }
    )

    return topup_tx


async def cleanup_test_user():
    """Clean up test data."""
    await CreditRefundRequest.prisma().delete_many(
        where={"userId": REFUND_TEST_USER_ID}
    )
    await CreditTransaction.prisma().delete_many(where={"userId": REFUND_TEST_USER_ID})
    await UserBalance.prisma().delete_many(where={"userId": REFUND_TEST_USER_ID})
    await User.prisma().delete_many(where={"id": REFUND_TEST_USER_ID})


@pytest.mark.asyncio(loop_scope="session")
async def test_deduct_credits_atomic(server: SpinTestServer):
    """Test that deduct_credits is atomic and creates transaction correctly."""
    topup_tx = await setup_test_user_with_topup()

    try:
        # Create a mock refund object
        refund = MagicMock(spec=stripe.Refund)
        refund.id = "re_test_refund_123"
        refund.payment_intent = topup_tx.transactionKey
        refund.amount = 500  # Refund $5 of the $10 top-up
        refund.status = "succeeded"
        refund.reason = "requested_by_customer"
        refund.created = int(datetime.now(timezone.utc).timestamp())

        # Create refund request record (simulating webhook flow)
        await CreditRefundRequest.prisma().create(
            data={
                "userId": REFUND_TEST_USER_ID,
                "amount": 500,
                "transactionKey": topup_tx.transactionKey,  # Should match the original transaction
                "reason": "Test refund",
            }
        )

        # Call deduct_credits
        await credit_system.deduct_credits(refund)

        # Verify the user's balance was deducted
        user_balance = await UserBalance.prisma().find_unique(
            where={"userId": REFUND_TEST_USER_ID}
        )
        assert user_balance is not None
        assert (
            user_balance.balance == 500
        ), f"Expected balance 500, got {user_balance.balance}"

        # Verify refund transaction was created
        refund_tx = await CreditTransaction.prisma().find_first(
            where={
                "userId": REFUND_TEST_USER_ID,
                "type": CreditTransactionType.REFUND,
                "transactionKey": refund.id,
            }
        )
        assert refund_tx is not None
        assert refund_tx.amount == -500
        assert refund_tx.runningBalance == 500
        assert refund_tx.isActive

        # Verify refund request was updated
        refund_request = await CreditRefundRequest.prisma().find_first(
            where={
                "userId": REFUND_TEST_USER_ID,
                "transactionKey": topup_tx.transactionKey,
            }
        )
        assert refund_request is not None
        assert (
            refund_request.result
            == "The refund request has been approved, the amount will be credited back to your account."
        )

    finally:
        await cleanup_test_user()


@pytest.mark.asyncio(loop_scope="session")
async def test_deduct_credits_user_not_found(server: SpinTestServer):
    """Test that deduct_credits raises error if transaction not found (which means user doesn't exist)."""
    # Create a mock refund object that references a non-existent payment intent
    refund = MagicMock(spec=stripe.Refund)
    refund.id = "re_test_refund_nonexistent"
    refund.payment_intent = "pi_test_nonexistent"  # This payment intent doesn't exist
    refund.amount = 500
    refund.status = "succeeded"
    refund.reason = "requested_by_customer"
    refund.created = int(datetime.now(timezone.utc).timestamp())

    # Should raise error for missing transaction
    with pytest.raises(Exception):  # Should raise NotFoundError for missing transaction
        await credit_system.deduct_credits(refund)


@pytest.mark.asyncio(loop_scope="session")
@patch("backend.data.credit.settings")
@patch("stripe.Dispute.modify")
@patch("backend.data.credit.get_user_by_id")
async def test_handle_dispute_with_sufficient_balance(
    mock_get_user, mock_stripe_modify, mock_settings, server: SpinTestServer
):
    """Test handling dispute when user has sufficient balance (dispute gets closed)."""
    topup_tx = await setup_test_user_with_topup()

    try:
        # Mock settings to have a low tolerance threshold
        mock_settings.config.refund_credit_tolerance_threshold = 0

        # Mock the user lookup
        mock_user = MagicMock()
        mock_user.email = f"{REFUND_TEST_USER_ID}@example.com"
        mock_get_user.return_value = mock_user

        # Create a mock dispute object for small amount (user has 1000, disputing 100)
        dispute = MagicMock(spec=stripe.Dispute)
        dispute.id = "dp_test_dispute_123"
        dispute.payment_intent = topup_tx.transactionKey
        dispute.amount = 100  # Small dispute amount
        dispute.status = "pending"
        dispute.reason = "fraudulent"
        dispute.created = int(datetime.now(timezone.utc).timestamp())

        # Mock the close method to prevent real API calls
        dispute.close = MagicMock()

        # Handle the dispute
        await credit_system.handle_dispute(dispute)

        # Verify dispute.close() was called (since user has sufficient balance)
        dispute.close.assert_called_once()

        # Verify no stripe evidence was added since dispute was closed
        mock_stripe_modify.assert_not_called()

        # Verify the user's balance was NOT deducted (dispute was closed)
        user_balance = await UserBalance.prisma().find_unique(
            where={"userId": REFUND_TEST_USER_ID}
        )
        assert user_balance is not None
        assert (
            user_balance.balance == 1000
        ), f"Balance should remain 1000, got {user_balance.balance}"

    finally:
        await cleanup_test_user()


@pytest.mark.asyncio(loop_scope="session")
@patch("backend.data.credit.settings")
@patch("stripe.Dispute.modify")
@patch("backend.data.credit.get_user_by_id")
async def test_handle_dispute_with_insufficient_balance(
    mock_get_user, mock_stripe_modify, mock_settings, server: SpinTestServer
):
    """Test handling dispute when user has insufficient balance (evidence gets added)."""
    topup_tx = await setup_test_user_with_topup()

    # Save original method for restoration before any try blocks
    original_get_history = credit_system.get_transaction_history

    try:
        # Mock settings to have a high tolerance threshold so dispute isn't closed
        mock_settings.config.refund_credit_tolerance_threshold = 2000

        # Mock the user lookup
        mock_user = MagicMock()
        mock_user.email = f"{REFUND_TEST_USER_ID}@example.com"
        mock_get_user.return_value = mock_user

        # Mock the transaction history method to return an async result
        from unittest.mock import AsyncMock

        mock_history = MagicMock()
        mock_history.transactions = []
        credit_system.get_transaction_history = AsyncMock(return_value=mock_history)

        # Create a mock dispute object for full amount (user has 1000, disputing 1000)
        dispute = MagicMock(spec=stripe.Dispute)
        dispute.id = "dp_test_dispute_pending"
        dispute.payment_intent = topup_tx.transactionKey
        dispute.amount = 1000
        dispute.status = "warning_needs_response"
        dispute.created = int(datetime.now(timezone.utc).timestamp())

        # Mock the close method to prevent real API calls
        dispute.close = MagicMock()

        # Handle the dispute (evidence should be added)
        await credit_system.handle_dispute(dispute)

        # Verify dispute.close() was NOT called (insufficient balance after tolerance)
        dispute.close.assert_not_called()

        # Verify stripe evidence was added since dispute wasn't closed
        mock_stripe_modify.assert_called_once()

        # Verify the user's balance was NOT deducted (handle_dispute doesn't deduct credits)
        user_balance = await UserBalance.prisma().find_unique(
            where={"userId": REFUND_TEST_USER_ID}
        )
        assert user_balance is not None
        assert user_balance.balance == 1000, "Balance should remain unchanged"

    finally:
        credit_system.get_transaction_history = original_get_history
        await cleanup_test_user()


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_refunds(server: SpinTestServer):
    """Test that concurrent refunds are handled atomically."""
    import asyncio

    topup_tx = await setup_test_user_with_topup()

    try:
        # Create multiple refund requests
        refund_requests = []
        for i in range(5):
            req = await CreditRefundRequest.prisma().create(
                data={
                    "userId": REFUND_TEST_USER_ID,
                    "amount": 100,  # $1 each
                    "transactionKey": topup_tx.transactionKey,
                    "reason": f"Test refund {i}",
                }
            )
            refund_requests.append(req)

        # Create refund tasks to run concurrently
        async def process_refund(index: int):
            refund = MagicMock(spec=stripe.Refund)
            refund.id = f"re_test_concurrent_{index}"
            refund.payment_intent = topup_tx.transactionKey
            refund.amount = 100  # $1 refund
            refund.status = "succeeded"
            refund.reason = "requested_by_customer"
            refund.created = int(datetime.now(timezone.utc).timestamp())

            try:
                await credit_system.deduct_credits(refund)
                return "success"
            except Exception as e:
                return f"error: {e}"

        # Run refunds concurrently
        results = await asyncio.gather(
            *[process_refund(i) for i in range(5)], return_exceptions=True
        )

        # All should succeed
        assert all(r == "success" for r in results), f"Some refunds failed: {results}"

        # Verify final balance - with non-atomic implementation, this will demonstrate race condition
        # EXPECTED BEHAVIOR: Due to race conditions, not all refunds will be properly processed
        # The balance will be incorrect (higher than expected) showing lost updates
        user_balance = await UserBalance.prisma().find_unique(
            where={"userId": REFUND_TEST_USER_ID}
        )
        assert user_balance is not None

        # With atomic implementation, this should be 500 (1000 - 5*100)
        # With current non-atomic implementation, this will likely be wrong due to race conditions
        print(f"DEBUG: Final balance = {user_balance.balance}, expected = 500")

        # With atomic implementation, all 5 refunds should process correctly
        assert (
            user_balance.balance == 500
        ), f"Expected balance 500 after 5 refunds of 100 each, got {user_balance.balance}"

        # Verify all refund transactions exist
        refund_txs = await CreditTransaction.prisma().find_many(
            where={
                "userId": REFUND_TEST_USER_ID,
                "type": CreditTransactionType.REFUND,
            }
        )
        assert (
            len(refund_txs) == 5
        ), f"Expected 5 refund transactions, got {len(refund_txs)}"

        running_balances: set[int] = {
            tx.runningBalance for tx in refund_txs if tx.runningBalance is not None
        }

        # Verify all balances are valid intermediate states
        for balance in running_balances:
            assert (
                500 <= balance <= 1000
            ), f"Invalid balance {balance}, should be between 500 and 1000"

        # Final balance should be present
        assert (
            500 in running_balances
        ), f"Final balance 500 should be in {running_balances}"

        # All balances should be unique and form a valid sequence
        sorted_balances = sorted(running_balances, reverse=True)
        assert (
            len(sorted_balances) == 5
        ), f"Expected 5 unique balances, got {len(sorted_balances)}"

    finally:
        await cleanup_test_user()
