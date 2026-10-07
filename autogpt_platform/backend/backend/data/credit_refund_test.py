"""
Tests for credit system refund and dispute operations.

These tests ensure that refund operations (deduct_credits, handle_dispute)
are atomic and maintain data consistency.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import CreditRefundRequestStatus, CreditTransactionType
from prisma.models import (
    ChatSession,
    CreditRefundRequest,
    CreditTransaction,
    User,
    UserBalance,
)

from backend.data import credit
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
async def test_deduct_credits_ignores_refund_of_non_topup_payment(
    server: SpinTestServer,
):
    """A refund of a payment that bought no credits, such as a subscription
    invoice, returns without touching the ledger or the user's refund requests."""
    topup_tx = await setup_test_user_with_topup()

    try:
        await CreditRefundRequest.prisma().create(
            data={
                "userId": REFUND_TEST_USER_ID,
                "amount": 500,
                "transactionKey": topup_tx.transactionKey,
                "reason": "Test refund",
            }
        )
        refund = MagicMock(spec=stripe.Refund)
        refund.id = "re_test_non_topup"
        refund.payment_intent = "pi_test_subscription_invoice"
        refund.amount = 6000
        refund.status = "succeeded"
        refund.reason = "requested_by_customer"

        await credit_system.deduct_credits(refund)

        assert (
            await CreditTransaction.prisma().count(where={"transactionKey": refund.id})
            == 0
        )
        user_balance = await UserBalance.prisma().find_unique(
            where={"userId": REFUND_TEST_USER_ID}
        )
        assert user_balance is not None
        assert user_balance.balance == 1000
        refund_request = await CreditRefundRequest.prisma().find_first(
            where={"userId": REFUND_TEST_USER_ID}
        )
        assert refund_request is not None
        assert refund_request.status == CreditRefundRequestStatus.PENDING
    finally:
        await cleanup_test_user()


@pytest.mark.asyncio(loop_scope="session")
@patch("backend.data.credit.settings")
@patch("stripe.Dispute.modify_async")
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
@patch("stripe.Dispute.modify_async")
@patch("backend.data.credit.get_user_by_id")
async def test_handle_dispute_with_insufficient_balance(
    mock_get_user, mock_stripe_modify, mock_settings, server: SpinTestServer
):
    """Test handling dispute when user has insufficient balance (evidence gets added).

    handle_dispute reads CreditTransaction rows directly via prisma, so the
    evidence-building path is exercised against the real DB state created by
    setup_test_user_with_topup — this is an integration test by design.
    """
    topup_tx = await setup_test_user_with_topup()

    try:
        # Mock settings to have a high tolerance threshold so dispute isn't closed
        mock_settings.config.refund_credit_tolerance_threshold = 2000

        # Mock the user lookup
        mock_user = MagicMock()
        mock_user.email = f"{REFUND_TEST_USER_ID}@example.com"
        mock_get_user.return_value = mock_user

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
        await cleanup_test_user()


@pytest.mark.asyncio(loop_scope="session")
@patch("backend.data.credit.queue_notification_async", new_callable=AsyncMock)
@patch("stripe.PaymentIntent.retrieve_async", new_callable=AsyncMock)
@patch("stripe.Dispute.modify_async")
async def test_handle_dispute_ignores_dispute_of_non_topup_payment(
    mock_stripe_modify, mock_retrieve, mock_notify, server: SpinTestServer
):
    """A dispute of a payment that bought neither credits nor a subscription is
    neither accepted nor contested, and the user's top-up is left as it was."""
    topup_tx = await setup_test_user_with_topup()
    mock_retrieve.return_value = stripe.PaymentIntent.construct_from(
        {"id": "pi_test_subscription_invoice", "invoice": None}, "sk_test"
    )

    try:
        dispute = MagicMock(spec=stripe.Dispute)
        dispute.id = "du_test_non_topup"
        dispute.payment_intent = "pi_test_subscription_invoice"
        dispute.amount = 6000
        dispute.status = "needs_response"
        dispute.close = MagicMock()

        await credit_system.handle_dispute(dispute)

        dispute.close.assert_not_called()
        mock_stripe_modify.assert_not_called()
        mock_notify.assert_not_awaited()
        assert await CreditTransaction.prisma().find_many(
            where={"userId": REFUND_TEST_USER_ID}
        ) == [topup_tx]
        user_balance = await UserBalance.prisma().find_unique(
            where={"userId": REFUND_TEST_USER_ID}
        )
        assert user_balance is not None
        assert user_balance.balance == 1000
    finally:
        await cleanup_test_user()


SUBSCRIPTION_CUSTOMER_ID = "cus_refund_test_user"


async def setup_subscriber_with_a_chat():
    await setup_test_user_with_topup()
    await User.prisma().update(
        where={"id": REFUND_TEST_USER_ID},
        data={"stripeCustomerId": SUBSCRIPTION_CUSTOMER_ID},
    )
    await ChatSession.prisma().create(data={"userId": REFUND_TEST_USER_ID})


def subscription_dispute(
    period_start: datetime, period_end: datetime
) -> tuple[stripe.Dispute, stripe.PaymentIntent, stripe.Subscription]:
    dispute = stripe.Dispute.construct_from(
        {
            "id": "du_test_subscription",
            "object": "dispute",
            "payment_intent": "pi_test_subscription_invoice",
            "amount": 5000,
            "reason": "fraudulent",
            "status": "needs_response",
            "evidence_details": {"due_by": int(period_end.timestamp())},
        },
        "sk_test",
    )
    payment_intent = stripe.PaymentIntent.construct_from(
        {
            "id": "pi_test_subscription_invoice",
            "object": "payment_intent",
            "invoice": {
                "id": "in_test_subscription",
                "object": "invoice",
                "customer": SUBSCRIPTION_CUSTOMER_ID,
                "subscription": "sub_test_pro",
                "lines": {
                    "object": "list",
                    "data": [
                        {
                            "object": "line_item",
                            "description": "1 × AutoGPT Pro (at $50.00 / month)",
                            "period": {
                                "start": int(period_start.timestamp()),
                                "end": int(period_end.timestamp()),
                            },
                        }
                    ],
                },
            },
        },
        "sk_test",
    )
    subscription = stripe.Subscription.construct_from(
        {
            "id": "sub_test_pro",
            "object": "subscription",
            "status": "active",
            "start_date": int((period_start - timedelta(days=60)).timestamp()),
        },
        "sk_test",
    )
    return dispute, payment_intent, subscription


@pytest.mark.asyncio(loop_scope="session")
@patch("backend.data.credit.queue_notification_async", new_callable=AsyncMock)
@patch("stripe.Subscription.retrieve_async", new_callable=AsyncMock)
@patch("stripe.PaymentIntent.retrieve_async", new_callable=AsyncMock)
@patch("stripe.Dispute.modify_async", new_callable=AsyncMock)
async def test_handle_dispute_of_subscription_payment_notifies_without_contesting(
    mock_modify, mock_retrieve, mock_subscription, mock_notify, server: SpinTestServer
):
    """Under the default settings the refunds team hears about a disputed
    subscription payment and nothing is submitted to Stripe on its behalf."""
    now = datetime.now(timezone.utc)
    start, end = now - timedelta(days=10), now + timedelta(days=20)
    await setup_subscriber_with_a_chat()
    dispute, payment_intent, subscription = subscription_dispute(start, end)
    mock_retrieve.return_value = payment_intent
    mock_subscription.return_value = subscription

    try:
        await credit_system.handle_dispute(dispute)

        mock_modify.assert_not_awaited()
        mock_notify.assert_awaited_once()
        data = mock_notify.await_args.args[0].data
        assert data.kind == "dispute"
        assert data.refund_request_id == "du_test_subscription"
        assert data.amount_cents == 5000
        assert data.reason == "fraudulent"
        assert data.user_id == REFUND_TEST_USER_ID
        assert data.invoice_id == "in_test_subscription"
        assert data.plan_label == "1 × AutoGPT Pro (at $50.00 / month)"
        assert data.period_label == (
            f"{credit._date_label(start)} to {credit._date_label(end)}"
        )
        assert data.usage_label == "1 AutoPilot chat, 0 workflow runs"
        assert not data.contested
    finally:
        await cleanup_test_user()


@pytest.mark.asyncio(loop_scope="session")
@patch("backend.data.credit.queue_notification_async", new_callable=AsyncMock)
@patch("stripe.Subscription.retrieve_async", new_callable=AsyncMock)
@patch("stripe.PaymentIntent.retrieve_async", new_callable=AsyncMock)
@patch("stripe.Dispute.modify_async", new_callable=AsyncMock)
async def test_handle_dispute_of_subscription_payment_contests_when_enabled(
    mock_modify, mock_retrieve, mock_subscription, mock_notify, server: SpinTestServer
):
    """With contesting on, the dispute gets evidence built from the subscription
    record, and the mail says it was sent."""
    now = datetime.now(timezone.utc)
    start, end = now - timedelta(days=10), now + timedelta(days=20)
    await setup_subscriber_with_a_chat()
    dispute, payment_intent, subscription = subscription_dispute(start, end)
    mock_retrieve.return_value = payment_intent
    mock_subscription.return_value = subscription

    try:
        with patch.object(
            credit.settings.config, "contest_subscription_disputes", True
        ):
            await credit_system.handle_dispute(dispute)

        mock_modify.assert_awaited_once()
        assert mock_modify.await_args.args == ("du_test_subscription",)
        evidence = mock_modify.await_args.kwargs["evidence"]
        assert evidence["customer_email_address"] == (
            f"{REFUND_TEST_USER_ID}@example.com"
        )
        assert evidence["service_date"] == credit._date_label(start)
        assert "1 AutoPilot chat, 0 workflow runs" in evidence["access_activity_log"]
        for fact in (
            "in_test_subscription",
            "sub_test_pro",
            "active",
            "1 × AutoGPT Pro (at $50.00 / month)",
            credit._date_label(end),
        ):
            assert fact in evidence["uncategorized_text"]
        assert mock_notify.await_args.args[0].data.contested
    finally:
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
