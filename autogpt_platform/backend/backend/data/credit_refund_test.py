"""
Tests for credit system refund and dispute operations.

These tests ensure that refund operations (deduct_credits, handle_dispute)
are atomic and maintain data consistency.
"""

import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import pytest
import stripe
from httpx import ASGITransport, AsyncClient
from prisma.enums import CreditRefundRequestStatus, CreditTransactionType
from prisma.models import CreditRefundRequest, CreditTransaction, User, UserBalance

from backend.api.features.billing.subscriptions.routes import router as billing_router
from backend.data import credit as credit_module
from backend.data.credit import UsageTransactionMetadata, UserCredit
from backend.util.exceptions import InsufficientBalanceError
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer

credit_system = UserCredit()
WEBHOOK_SECRET = "whsec_refund_update_test_only"
webhook_app = fastapi.FastAPI()
webhook_app.include_router(billing_router)

# Test user ID for refund tests
REFUND_TEST_USER_ID = "refund-test-user"


async def setup_test_user_with_topup(charged_cents: int | None = None):
    """Create a test user with initial balance and a top-up transaction, whose
    PaymentIntent charged `charged_cents` for its 1000 credits when given."""
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
            "metadata": SafeJson(
                {"payment_intent": {"id": "pi_test_12345", "amount": charged_cents}}
                if charged_cents
                else {"stripe_payment_intent": "pi_test_12345"}
            ),
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
async def test_deduct_credits_ignores_refund_of_subscription_invoice(
    subscription_payment_intent, server: SpinTestServer
):
    """A refund of a subscription invoice, which bought no credits, returns
    without touching the ledger or the user's refund requests."""
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

        assert subscription_payment_intent.await_args.args == (
            "pi_test_subscription_invoice",
        )
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
        dispute.status = "needs_response"
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

        # The disputed credits are taken as the dispute is accepted, and its
        # `charge.dispute.closed` webhook does not take them again.
        assert await credit_system.get_credits(REFUND_TEST_USER_ID) == 900
        dispute.status = "lost"
        with patch.object(UserCredit, "_send_refund_notification", AsyncMock()):
            await credit_system.deduct_credits(dispute)
        assert await credit_system.get_credits(REFUND_TEST_USER_ID) == 900

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
        dispute.status = "needs_response"
        dispute.created = int(datetime.now(timezone.utc).timestamp())

        # Mock the close method to prevent real API calls
        dispute.close = MagicMock()

        # Handle the dispute (evidence should be added)
        await credit_system.handle_dispute(dispute)

        # Verify dispute.close() was NOT called (insufficient balance after tolerance)
        dispute.close.assert_not_called()

        # Verify stripe evidence was added since dispute wasn't closed
        mock_stripe_modify.assert_called_once()

        # A contested dispute takes no credits.
        user_balance = await UserBalance.prisma().find_unique(
            where={"userId": REFUND_TEST_USER_ID}
        )
        assert user_balance is not None
        assert user_balance.balance == 1000, "Balance should remain unchanged"

    finally:
        await cleanup_test_user()


@pytest.mark.asyncio(loop_scope="session")
@patch("stripe.Dispute.modify_async")
async def test_handle_dispute_ignores_dispute_of_subscription_invoice(
    mock_stripe_modify, subscription_payment_intent, server: SpinTestServer
):
    """A dispute of a subscription invoice is neither accepted nor contested,
    and the user's unrelated top-up is left as it was."""
    topup_tx = await setup_test_user_with_topup()

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


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "request_type, request_id, status",
    [
        (stripe.Refund, "re_test_early", "succeeded"),
        (stripe.Dispute, "du_test_early", "lost"),
    ],
    ids=["refund", "lost-dispute"],
)
async def test_refund_or_lost_dispute_before_checkout_fulfillment_deducts_on_retry(
    request_type, request_id, status, server: SpinTestServer
):
    """A refund or lost dispute that arrives while its Checkout top-up is still
    unfulfilled raises, so Stripe retries it, and deducts once fulfillment ran."""
    await setup_test_user_with_topup()
    try:
        await CreditTransaction.prisma().create(
            data={
                "userId": REFUND_TEST_USER_ID,
                "amount": 2000,
                "type": CreditTransactionType.TOP_UP,
                "transactionKey": "cs_test_unfulfilled",
                "isActive": False,
            }
        )
        request = MagicMock(spec=request_type)
        request.id = request_id
        request.payment_intent = "pi_test_checkout"
        request.amount = 2000
        request.status = status
        request.reason = "requested_by_customer"
        paid_session = stripe.checkout.Session.construct_from(
            {
                "id": "cs_test_unfulfilled",
                "object": "checkout.session",
                "payment_status": "paid",
                "amount_total": 2000,
                "currency": "usd",
                "payment_intent": {
                    "id": "pi_test_checkout",
                    "object": "payment_intent",
                },
            },
            "sk_test",
        )

        with (
            patch(
                "stripe.PaymentIntent.retrieve_async",
                new=_stripe_payment_intents(invoice=None),
            ),
            patch(
                "stripe.checkout.Session.retrieve_async",
                new_callable=AsyncMock,
                return_value=paid_session,
            ),
        ):
            with pytest.raises(RuntimeError):
                await credit_system.deduct_credits(request)
            assert (
                await CreditTransaction.prisma().count(
                    where={"transactionKey": request_id}
                )
                == 0
            )

            await credit_system.fulfill_checkout(session_id="cs_test_unfulfilled")
            await credit_system.deduct_credits(request)

        deduction = await CreditTransaction.prisma().find_first(
            where={"transactionKey": request_id}
        )
        assert deduction is not None
        assert deduction.amount == -2000
        user_balance = await UserBalance.prisma().find_unique(
            where={"userId": REFUND_TEST_USER_ID}
        )
        assert user_balance is not None
        assert user_balance.balance == 1000
    finally:
        await cleanup_test_user()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "grant_on, lookup_error, error",
    [
        (True, None, RuntimeError),
        (False, stripe.APIConnectionError("down"), stripe.StripeError),
    ],
    ids=["invoices-grant-credits", "lookup-fails"],
)
async def test_refund_without_a_top_up_raises_unless_shown_creditless(
    grant_on, lookup_error, error, monkeypatch, server: SpinTestServer
):
    """A refund matching no top-up is acknowledged only on proof it bought no
    credits: an invoice bought some when invoices grant credits, and a failed
    lookup proves nothing."""
    monkeypatch.setattr(
        credit_module.settings.config, "enable_subscription_credit_grant", grant_on
    )
    refund = MagicMock(spec=stripe.Refund)
    refund.id = "re_test_unproven"
    refund.payment_intent = "pi_test_subscription_invoice"
    refund.amount = 6000
    refund.status = "succeeded"

    retrieve = _stripe_payment_intents(invoice="in_test_subscription")
    if lookup_error:
        retrieve.side_effect = lookup_error

    with (
        patch("stripe.PaymentIntent.retrieve_async", new=retrieve),
        pytest.raises(error),
    ):
        await credit_system.deduct_credits(refund)

    assert (
        await CreditTransaction.prisma().count(where={"transactionKey": refund.id}) == 0
    )


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "request_type, status",
    [(stripe.Refund, "succeeded"), (stripe.Dispute, "lost")],
    ids=["refund", "lost-dispute"],
)
async def test_refund_or_lost_dispute_without_a_payment_intent_raises(
    request_type, status, server: SpinTestServer
):
    """A refund or lost dispute with no PaymentIntent, as on a charge made
    outside one, cannot be shown creditless: it raises, so Stripe retries,
    without a lookup of a PaymentIntent named "None"."""
    request = MagicMock(spec=request_type)
    request.id = "re_test_no_intent"
    request.payment_intent = None
    request.amount = 6000
    request.status = status
    retrieve = _stripe_payment_intents(invoice="in_test_subscription")

    with (
        patch("stripe.PaymentIntent.retrieve_async", new=retrieve),
        pytest.raises(RuntimeError),
    ):
        await credit_system.deduct_credits(request)

    retrieve.assert_not_awaited()


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


@pytest.mark.asyncio(loop_scope="session")
@patch("stripe.Refund.create_async", new_callable=AsyncMock)
async def test_top_up_refund_takes_credits_on_approval(
    mock_refund_create, server: SpinTestServer
):
    """An auto-approved refund takes its credits before Stripe's webhook arrives."""
    topup_tx = await setup_test_user_with_topup()
    try:
        mock_refund_create.return_value = _stripe_refund(topup_tx.transactionKey, 1000)

        refunded = await credit_system.top_up_refund(
            REFUND_TEST_USER_ID, topup_tx.transactionKey, {}
        )

        assert refunded == 1000
        with pytest.raises(InsufficientBalanceError):
            await credit_system.spend_credits(
                REFUND_TEST_USER_ID, 1, UsageTransactionMetadata(reason="Test spend")
            )
    finally:
        await cleanup_test_user()


@pytest.mark.asyncio(loop_scope="session")
@patch("stripe.Refund.create_async", new_callable=AsyncMock)
async def test_top_up_refund_returns_credits_when_stripe_fails(
    mock_refund_create, server: SpinTestServer
):
    """Credits taken on approval come back when Stripe refunds nothing, and a
    later refund from the Stripe dashboard is still deducted."""
    topup_tx = await setup_test_user_with_topup()
    try:
        mock_refund_create.side_effect = stripe.APIConnectionError("Stripe is down")

        with pytest.raises(stripe.APIConnectionError):
            await credit_system.top_up_refund(
                REFUND_TEST_USER_ID, topup_tx.transactionKey, {}
            )
        assert await credit_system.get_credits(REFUND_TEST_USER_ID) == 1000

        await credit_system.deduct_credits(
            _stripe_refund(topup_tx.transactionKey, 1000)
        )
        assert await credit_system.get_credits(REFUND_TEST_USER_ID) == 0
    finally:
        await cleanup_test_user()


@pytest.mark.parametrize(
    "balance, approved, balance_after",
    [(700, True, -300), (400, False, 400)],
    ids=["within-tolerance", "beyond-tolerance"],
)
@pytest.mark.asyncio(loop_scope="session")
@patch("stripe.Refund.create_async", new_callable=AsyncMock)
async def test_top_up_refund_auto_approves_only_within_the_tolerance(
    mock_refund_create,
    balance: int,
    approved: bool,
    balance_after: int,
    monkeypatch,
    server: SpinTestServer,
):
    """A refund whose credits the balance covers within the tolerance is approved
    and taken; any other goes to manual review and takes nothing."""
    topup_tx = await setup_test_user_with_topup()
    monkeypatch.setattr(
        credit_module.settings.config, "refund_credit_tolerance_threshold", 500
    )
    mock_refund_create.return_value = _stripe_refund(topup_tx.transactionKey, 1000)
    notify = AsyncMock()
    try:
        await UserBalance.prisma().update(
            where={"userId": REFUND_TEST_USER_ID}, data={"balance": balance}
        )
        with patch.object(UserCredit, "_send_refund_notification", notify):
            refunded = await credit_system.top_up_refund(
                REFUND_TEST_USER_ID, topup_tx.transactionKey, {}
            )

        assert refunded == (1000 if approved else 0)
        assert mock_refund_create.await_count == approved
        assert await credit_system.get_credits(REFUND_TEST_USER_ID) == balance_after
        assert (
            await CreditTransaction.prisma().count(
                where={
                    "userId": REFUND_TEST_USER_ID,
                    "type": CreditTransactionType.REFUND,
                }
            )
            == approved
        )
        assert [call.kwargs["kind"] for call in notify.await_args_list] == (
            [] if approved else ["request"]
        )
    finally:
        await cleanup_test_user()


@pytest.mark.asyncio(loop_scope="session")
@patch("stripe.Refund.create_async", new_callable=AsyncMock)
async def test_top_up_refund_keeps_credits_the_webhook_settled(
    mock_refund_create, server: SpinTestServer
):
    """When Stripe's reply fails after the refund's webhook settled it, the
    refund was made, so its credits stay taken."""
    topup_tx = await setup_test_user_with_topup()
    refund = _stripe_refund(topup_tx.transactionKey, 1000)

    async def refund_then_fail(**_):
        await credit_system.deduct_credits(refund)
        raise stripe.APIConnectionError("Read timed out")

    mock_refund_create.side_effect = refund_then_fail
    try:
        with pytest.raises(stripe.APIConnectionError):
            await credit_system.top_up_refund(
                REFUND_TEST_USER_ID, topup_tx.transactionKey, {}
            )

        assert await credit_system.get_credits(REFUND_TEST_USER_ID) == 0
        refund_txs = await CreditTransaction.prisma().find_many(
            where={
                "userId": REFUND_TEST_USER_ID,
                "type": CreditTransactionType.REFUND,
            }
        )
        assert [(tx.transactionKey, tx.amount) for tx in refund_txs] == [
            (refund.id, -1000)
        ]
    finally:
        await cleanup_test_user()


@pytest.mark.parametrize(
    "dashboard_first", [True, False], ids=["before-hold", "during-hold"]
)
@pytest.mark.asyncio(loop_scope="session")
@patch("stripe.Refund.create_async", new_callable=AsyncMock)
async def test_two_refunds_of_one_top_up_take_what_each_paid_back(
    mock_refund_create, dashboard_first: bool, monkeypatch, server: SpinTestServer
):
    """A part refunded from the Stripe dashboard and the user's refund of the
    rest take the top-up back once between them, in either order."""
    topup_tx = await setup_test_user_with_topup()
    monkeypatch.setattr(
        credit_module.settings.config, "refund_credit_tolerance_threshold", 500
    )
    dashboard_refund = _stripe_refund(topup_tx.transactionKey, 300)
    dashboard_refund.id = "re_test_dashboard"
    user_refund = _stripe_refund(topup_tx.transactionKey, 700)

    async def create_refund(**_):
        if not dashboard_first:
            await credit_system.deduct_credits(dashboard_refund)
        return user_refund

    mock_refund_create.side_effect = create_refund
    try:
        if dashboard_first:
            await credit_system.deduct_credits(dashboard_refund)
        await credit_system.top_up_refund(
            REFUND_TEST_USER_ID, topup_tx.transactionKey, {}
        )
        await credit_system.deduct_credits(user_refund)
        assert await credit_system.get_credits(REFUND_TEST_USER_ID) == 0

        await credit_system.deduct_credits(dashboard_refund)
        await credit_system.deduct_credits(user_refund)
        assert await credit_system.get_credits(REFUND_TEST_USER_ID) == 0
    finally:
        await cleanup_test_user()


@pytest.mark.asyncio(loop_scope="session")
@patch("stripe.Refund.create_async", new_callable=AsyncMock)
async def test_a_retried_refund_returns_what_its_hold_took_beyond_it(
    mock_refund_create, monkeypatch, server: SpinTestServer
):
    """A refund whose return of the hold's surplus failed returns it when
    Stripe delivers the refund again."""
    topup_tx = await setup_test_user_with_topup()
    monkeypatch.setattr(
        credit_module.settings.config, "refund_credit_tolerance_threshold", 500
    )
    dashboard_refund = _stripe_refund(topup_tx.transactionKey, 300)
    dashboard_refund.id = "re_test_dashboard"
    user_refund = _stripe_refund(topup_tx.transactionKey, 700)
    mock_refund_create.return_value = user_refund
    add_transaction = UserCredit._add_transaction
    failures: list[str] = []

    async def fail_first_surplus(self, *args, **kwargs):
        key = str(kwargs.get("transaction_key"))
        if key.endswith("-UNHELD") and not failures:
            failures.append(key)
            raise RuntimeError("Database connection lost")
        return await add_transaction(self, *args, **kwargs)

    try:
        await credit_system.deduct_credits(dashboard_refund)
        await credit_system.top_up_refund(
            REFUND_TEST_USER_ID, topup_tx.transactionKey, {}
        )
        with patch.object(UserCredit, "_add_transaction", fail_first_surplus):
            with pytest.raises(RuntimeError):
                await credit_system.deduct_credits(user_refund)
            await credit_system.deduct_credits(user_refund)

        assert failures == [f"{user_refund.id}-UNHELD"]
        assert await credit_system.get_credits(REFUND_TEST_USER_ID) == 0
    finally:
        await cleanup_test_user()


@pytest.mark.parametrize("charged_cents", [1100, 600], ids=["tax", "promotion-code"])
@pytest.mark.asyncio(loop_scope="session")
async def test_deduct_credits_converts_the_charge_to_credits(
    charged_cents: int, server: SpinTestServer
):
    """A refund of the whole charge takes back the whole top-up, and no more,
    whatever tax or a promotion code made the charge."""
    topup_tx = await setup_test_user_with_topup(charged_cents=charged_cents)
    try:
        await credit_system.deduct_credits(
            _stripe_refund(topup_tx.transactionKey, charged_cents)
        )
        assert await credit_system.get_credits(REFUND_TEST_USER_ID) == 0

        over_refund = _stripe_refund(topup_tx.transactionKey, charged_cents + 1)
        over_refund.id = "re_test_over_refund"
        with pytest.raises(AssertionError):
            await credit_system.deduct_credits(over_refund)
    finally:
        await cleanup_test_user()


@pytest.mark.asyncio(loop_scope="session")
@patch("stripe.Dispute.modify_async")
@patch("backend.data.credit.get_user_by_id")
async def test_handle_dispute_weighs_credits_not_the_charge(
    mock_get_user, mock_stripe_modify, monkeypatch, server: SpinTestServer
):
    """A dispute of a discounted top-up is weighed against its 1000 credits, not
    the 600 cents charged, so a balance of 700 contests it."""
    topup_tx = await setup_test_user_with_topup(charged_cents=600)
    await UserBalance.prisma().update(
        where={"userId": REFUND_TEST_USER_ID}, data={"balance": 700}
    )
    monkeypatch.setattr(
        credit_module.settings.config, "refund_credit_tolerance_threshold", 0
    )
    mock_get_user.return_value = MagicMock(email=f"{REFUND_TEST_USER_ID}@example.com")
    try:
        dispute = MagicMock(spec=stripe.Dispute)
        dispute.id = "dp_test_dispute_discounted"
        dispute.payment_intent = topup_tx.transactionKey
        dispute.amount = 600
        dispute.status = "needs_response"
        dispute.close = MagicMock()

        await credit_system.handle_dispute(dispute)

        dispute.close.assert_not_called()
        mock_stripe_modify.assert_called_once()
        assert await credit_system.get_credits(REFUND_TEST_USER_ID) == 700
    finally:
        await cleanup_test_user()


@pytest.mark.parametrize(
    "dispute_amount, balance_after",
    [(600, 400), (100, 900)],
    ids=["balance-short-on-repeat", "balance-covers-repeat"],
)
@pytest.mark.asyncio(loop_scope="session")
@patch("stripe.Dispute.modify_async")
@patch("backend.data.credit.get_user_by_id")
async def test_handle_dispute_repeat_delivery_accepts_what_it_took(
    mock_get_user,
    mock_stripe_modify,
    dispute_amount: int,
    balance_after: int,
    monkeypatch,
    server: SpinTestServer,
):
    """A repeat delivery of an accepted chargeback closes it again and takes
    nothing more, whether or not the balance left still covers it."""
    topup_tx = await setup_test_user_with_topup()
    monkeypatch.setattr(
        credit_module.settings.config, "refund_credit_tolerance_threshold", 0
    )
    mock_get_user.return_value = MagicMock(email=f"{REFUND_TEST_USER_ID}@example.com")
    try:
        dispute = MagicMock(spec=stripe.Dispute)
        dispute.id = "dp_test_dispute_repeat"
        dispute.payment_intent = topup_tx.transactionKey
        dispute.amount = dispute_amount
        dispute.status = "needs_response"
        dispute.close = MagicMock()

        await credit_system.handle_dispute(dispute)
        await credit_system.handle_dispute(dispute)

        assert dispute.close.call_count == 2
        mock_stripe_modify.assert_not_called()
        assert await credit_system.get_credits(REFUND_TEST_USER_ID) == balance_after
    finally:
        await cleanup_test_user()


@pytest.mark.parametrize(
    "balance, accepted", [(1000, True), (700, False)], ids=["covered", "not-covered"]
)
@pytest.mark.asyncio(loop_scope="session")
@patch("stripe.Dispute.modify_async")
@patch("backend.data.credit.get_user_by_id")
async def test_handle_dispute_takes_no_credits_for_an_inquiry(
    mock_get_user,
    mock_stripe_modify,
    balance: int,
    accepted: bool,
    monkeypatch,
    server: SpinTestServer,
):
    """An inquiry is closed when the balance covers its credits and contested
    otherwise, and takes no credits either way."""
    topup_tx = await setup_test_user_with_topup(charged_cents=600)
    monkeypatch.setattr(
        credit_module.settings.config, "refund_credit_tolerance_threshold", 0
    )
    mock_get_user.return_value = MagicMock(email=f"{REFUND_TEST_USER_ID}@example.com")
    try:
        await UserBalance.prisma().update(
            where={"userId": REFUND_TEST_USER_ID}, data={"balance": balance}
        )
        dispute = MagicMock(spec=stripe.Dispute)
        dispute.id = "dp_test_inquiry"
        dispute.payment_intent = topup_tx.transactionKey
        dispute.amount = 600
        dispute.status = "warning_needs_response"
        dispute.close = MagicMock()

        await credit_system.handle_dispute(dispute)

        assert dispute.close.called == accepted
        assert mock_stripe_modify.called != accepted
        assert await credit_system.get_credits(REFUND_TEST_USER_ID) == balance
        assert (
            await CreditTransaction.prisma().count(
                where={
                    "userId": REFUND_TEST_USER_ID,
                    "type": CreditTransactionType.REFUND,
                }
            )
            == 0
        )
    finally:
        await cleanup_test_user()


@pytest.mark.parametrize("charged_cents", [1100, 600], ids=["tax", "promotion-code"])
@pytest.mark.parametrize(
    "webhook_first", [False, True], ids=["reply-first", "webhook-first"]
)
@pytest.mark.asyncio(loop_scope="session")
@patch("stripe.Refund.create_async", new_callable=AsyncMock)
async def test_auto_approved_refund_is_deducted_once(
    mock_refund_create,
    webhook_first: bool,
    charged_cents: int,
    server: SpinTestServer,
):
    """An auto-approved refund of a taxed or discounted top-up takes its credits
    once, whether Stripe's reply or its `refund.created` webhook comes first."""
    topup_tx = await setup_test_user_with_topup(charged_cents=charged_cents)
    refund = _stripe_refund(topup_tx.transactionKey, charged_cents)

    async def create_refund(**_):
        if webhook_first:
            await credit_system.deduct_credits(refund)
        return refund

    mock_refund_create.side_effect = create_refund
    try:
        await credit_system.top_up_refund(
            REFUND_TEST_USER_ID, topup_tx.transactionKey, {}
        )
        if not webhook_first:
            await credit_system.deduct_credits(refund)

        assert await credit_system.get_credits(REFUND_TEST_USER_ID) == 0
        refund_txs = await CreditTransaction.prisma().find_many(
            where={
                "userId": REFUND_TEST_USER_ID,
                "type": CreditTransactionType.REFUND,
            }
        )
        assert [(tx.transactionKey, tx.amount) for tx in refund_txs] == [
            (refund.id, -1000)
        ]
        refund_request = await CreditRefundRequest.prisma().find_first_or_raise(
            where={"userId": REFUND_TEST_USER_ID}
        )
        assert refund_request.amount == 1000
    finally:
        await cleanup_test_user()


@pytest.mark.parametrize(
    "balance, balance_after_request, balance_after",
    [(1000, 0, 0), (300, 300, -700)],
    ids=["auto-approved", "manual-review"],
)
@pytest.mark.asyncio(loop_scope="session")
@patch("stripe.Refund.create_async", new_callable=AsyncMock)
async def test_a_refund_reported_succeeded_by_an_update_is_settled_once(
    mock_refund_create,
    balance: int,
    balance_after_request: int,
    balance_after: int,
    monkeypatch,
    server: SpinTestServer,
):
    """A refund Stripe creates `pending` takes its credits and approves its
    request when `refund.updated` reports it succeeded; an update that leaves
    the status alone changes nothing."""
    topup_tx = await setup_test_user_with_topup()
    monkeypatch.setattr(
        credit_module.settings.config, "refund_credit_tolerance_threshold", 500
    )
    refund = {
        "object": "refund",
        "id": "re_test_settles_later",
        "payment_intent": topup_tx.transactionKey,
        "amount": 1000,
        "status": "pending",
        "reason": "requested_by_customer",
    }
    mock_refund_create.return_value = _stripe_refund(topup_tx.transactionKey, 1000)
    notify = AsyncMock()
    try:
        await UserBalance.prisma().update(
            where={"userId": REFUND_TEST_USER_ID}, data={"balance": balance}
        )
        with patch.object(UserCredit, "_send_refund_notification", notify):
            await credit_system.top_up_refund(
                REFUND_TEST_USER_ID, topup_tx.transactionKey, {}
            )
            await _deliver_refund_event("refund.created", refund)
            assert await credit_system.get_credits(REFUND_TEST_USER_ID) == (
                balance_after_request
            )

            await _deliver_refund_event(
                "refund.updated",
                {**refund, "status": "succeeded"},
                previous_attributes={"status": "pending"},
            )
            await _deliver_refund_event(
                "refund.updated",
                {**refund, "status": "succeeded"},
                previous_attributes={"destination_details": {"card": {}}},
            )

        assert await credit_system.get_credits(REFUND_TEST_USER_ID) == balance_after
        refund_txs = await CreditTransaction.prisma().find_many(
            where={
                "userId": REFUND_TEST_USER_ID,
                "type": CreditTransactionType.REFUND,
            }
        )
        assert [(tx.transactionKey, tx.amount) for tx in refund_txs] == [
            (refund["id"], -1000)
        ]
        refund_request = await CreditRefundRequest.prisma().find_first_or_raise(
            where={"userId": REFUND_TEST_USER_ID}
        )
        assert refund_request.status == CreditRefundRequestStatus.APPROVED
        assert [call.kwargs["kind"] for call in notify.await_args_list] == (
            ["processed"] if balance_after_request == 0 else ["request", "processed"]
        )
    finally:
        await cleanup_test_user()


@pytest.fixture
def subscription_payment_intent(monkeypatch):
    """A PaymentIntent that paid a subscription invoice, with the SDK's default
    API version moved past 2025-03-31.basil: only a request pinned before it
    sees the invoice."""
    monkeypatch.setattr(stripe, "api_version", "2026-09-30.endive")
    retrieve = _stripe_payment_intents(invoice="in_test_subscription")
    with patch("stripe.PaymentIntent.retrieve_async", new=retrieve):
        yield retrieve


def _stripe_payment_intents(invoice: str | None) -> AsyncMock:
    """Answers as Stripe does: an unknown id is a 404, and only API versions
    before 2025-03-31.basil name the invoice a PaymentIntent paid."""

    async def retrieve(payment_intent_id: str, stripe_version: str | None = None):
        if not payment_intent_id.startswith("pi_"):
            raise stripe.InvalidRequestError(
                f"No such payment_intent: '{payment_intent_id}'",
                "intent",
                code="resource_missing",
                http_status=404,
            )
        fields: dict = {"id": payment_intent_id, "object": "payment_intent"}
        if (stripe_version or stripe.api_version) < "2025-03-31":
            fields["invoice"] = invoice
        return stripe.PaymentIntent.construct_from(fields, "sk_test")

    return AsyncMock(side_effect=retrieve)


def _stripe_refund(payment_intent: str, amount: int) -> MagicMock:
    refund = MagicMock(spec=stripe.Refund)
    refund.id = f"re_{payment_intent}"
    refund.payment_intent = payment_intent
    refund.amount = amount
    refund.status = "succeeded"
    refund.reason = "requested_by_customer"
    return refund


async def _deliver_refund_event(
    event_type: str, refund: dict, previous_attributes: dict | None = None
) -> None:
    """Posts a signed Stripe event for `refund` to the billing webhook."""
    data: dict = {"object": refund}
    if previous_attributes is not None:
        data["previous_attributes"] = previous_attributes
    payload = json.dumps(
        {
            "id": f"evt_test_{time.time_ns()}",
            "object": "event",
            "type": event_type,
            "data": data,
        }
    )
    timestamp = int(time.time())
    signature = hmac.new(
        WEBHOOK_SECRET.encode(), f"{timestamp}.{payload}".encode(), hashlib.sha256
    ).hexdigest()
    routes = "backend.api.features.billing.subscriptions.routes"
    with (
        patch(f"{routes}.settings.secrets.stripe_webhook_secret", new=WEBHOOK_SECRET),
        patch(f"{routes}._claim_stripe_event", AsyncMock(return_value=True)),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=webhook_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/credits/stripe_webhook",
                content=payload,
                headers={"stripe-signature": f"t={timestamp},v1={signature}"},
            )
    assert response.status_code == 200
