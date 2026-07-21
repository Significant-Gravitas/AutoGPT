"""
Tests for credit system metadata handling to ensure JSON casting works correctly.

This test verifies that metadata parameters are properly serialized when passed
to raw SQL queries with JSONB columns.
"""

# type: ignore

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import CreditTransactionType, SubscriptionTier
from prisma.models import CreditTransaction, UserBalance

from backend.data.credit import (
    UserCredit,
    _datafast_metadata,
    create_subscription_checkout,
)
from backend.data.user import DEFAULT_USER_ID
from backend.util.json import SafeJson


@pytest.fixture
async def setup_test_user():
    """Setup test user and cleanup after test."""
    user_id = DEFAULT_USER_ID

    # Cleanup before test
    await CreditTransaction.prisma().delete_many(where={"userId": user_id})
    await UserBalance.prisma().delete_many(where={"userId": user_id})

    yield user_id

    # Cleanup after test
    await CreditTransaction.prisma().delete_many(where={"userId": user_id})
    await UserBalance.prisma().delete_many(where={"userId": user_id})


@pytest.mark.asyncio(loop_scope="session")
async def test_metadata_json_serialization(setup_test_user):
    """Test that metadata is properly serialized for JSONB column in raw SQL."""
    user_id = setup_test_user
    credit_system = UserCredit()

    # Test with complex metadata that would fail if not properly serialized
    complex_metadata = SafeJson(
        {
            "graph_exec_id": "test-12345",
            "reason": "Testing metadata serialization",
            "nested_data": {
                "key1": "value1",
                "key2": ["array", "of", "values"],
                "key3": {"deeply": {"nested": "object"}},
            },
            "special_chars": "Testing 'quotes' and \"double quotes\" and unicode: 🚀",
        }
    )

    # This should work without throwing a JSONB casting error
    balance, tx_key = await credit_system._add_transaction(
        user_id=user_id,
        amount=500,  # $5 top-up
        transaction_type=CreditTransactionType.TOP_UP,
        metadata=complex_metadata,
        is_active=True,
    )

    # Verify the transaction was created successfully
    assert balance == 500

    # Verify the metadata was stored correctly in the database
    transaction = await CreditTransaction.prisma().find_first(
        where={"userId": user_id, "transactionKey": tx_key}
    )

    assert transaction is not None
    assert transaction.metadata is not None

    # Verify the metadata contains our complex data
    metadata_dict: dict[str, Any] = dict(transaction.metadata)  # type: ignore
    assert metadata_dict["graph_exec_id"] == "test-12345"
    assert metadata_dict["reason"] == "Testing metadata serialization"
    assert metadata_dict["nested_data"]["key1"] == "value1"
    assert metadata_dict["nested_data"]["key3"]["deeply"]["nested"] == "object"
    assert (
        metadata_dict["special_chars"]
        == "Testing 'quotes' and \"double quotes\" and unicode: 🚀"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_enable_transaction_metadata_serialization(setup_test_user):
    """Test that _enable_transaction also handles metadata JSON serialization correctly."""
    user_id = setup_test_user
    credit_system = UserCredit()

    # First create an inactive transaction
    balance, tx_key = await credit_system._add_transaction(
        user_id=user_id,
        amount=300,
        transaction_type=CreditTransactionType.TOP_UP,
        metadata=SafeJson({"initial": "inactive_transaction"}),
        is_active=False,  # Create as inactive
    )

    # Initial balance should be 0 because transaction is inactive
    assert balance == 0

    # Now enable the transaction with new metadata
    enable_metadata = SafeJson(
        {
            "payment_method": "stripe",
            "payment_intent": "pi_test_12345",
            "activation_reason": "Payment confirmed",
            "complex_data": {"array": [1, 2, 3], "boolean": True, "null_value": None},
        }
    )

    # This should work without JSONB casting errors
    final_balance = await credit_system._enable_transaction(
        transaction_key=tx_key,
        user_id=user_id,
        metadata=enable_metadata,
    )

    # Now balance should reflect the activated transaction
    assert final_balance == 300

    # Verify the metadata was updated correctly
    transaction = await CreditTransaction.prisma().find_first(
        where={"userId": user_id, "transactionKey": tx_key}
    )

    assert transaction is not None
    assert transaction.isActive is True

    # Verify the metadata was updated with enable_metadata
    metadata_dict: dict[str, Any] = dict(transaction.metadata)  # type: ignore
    assert metadata_dict["payment_method"] == "stripe"
    assert metadata_dict["payment_intent"] == "pi_test_12345"
    assert metadata_dict["complex_data"]["array"] == [1, 2, 3]
    assert metadata_dict["complex_data"]["boolean"] is True
    assert metadata_dict["complex_data"]["null_value"] is None


def test_datafast_metadata_includes_only_present_ids():
    assert _datafast_metadata("vis_1", "ses_1") == {
        "datafast_visitor_id": "vis_1",
        "datafast_session_id": "ses_1",
    }


def test_datafast_metadata_drops_none_values():
    assert _datafast_metadata("vis_1", None) == {"datafast_visitor_id": "vis_1"}
    assert _datafast_metadata(None, None) == {}


def test_datafast_metadata_drops_invalid_values():
    # Oversized values are dropped entirely (a truncated ID is useless for
    # attribution), not truncated.
    assert _datafast_metadata("v" * 600, None) == {}
    # Whitespace is stripped; values that are blank or contain control
    # characters are dropped so they can never break Checkout.
    assert _datafast_metadata("  vis_1  ", "\x00bad") == {
        "datafast_visitor_id": "vis_1"
    }
    assert _datafast_metadata("   ", "ses_1") == {"datafast_session_id": "ses_1"}


@pytest.mark.asyncio
async def test_top_up_intent_attaches_datafast_metadata():
    """top_up_intent threads DataFast IDs into the Checkout session metadata."""
    mock_session = MagicMock()
    mock_session.id = "cs_test_topup"
    mock_session.url = "https://checkout.stripe.com/c/cs_test_topup"
    create_mock = MagicMock(return_value=mock_session)
    credit_system = UserCredit()
    with (
        patch(
            "backend.data.credit.get_stripe_customer_id",
            new_callable=AsyncMock,
            return_value="cus_123",
        ),
        patch(
            "backend.data.credit.get_feature_flag_value",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "backend.data.credit.stripe.checkout.Session.create",
            new=create_mock,
        ),
        patch.object(credit_system, "_add_transaction", new_callable=AsyncMock),
    ):
        await credit_system.top_up_intent(
            user_id="user-1",
            amount=500,
            datafast_visitor_id="vis_1",
            datafast_session_id="ses_1",
        )

    assert create_mock.call_args.kwargs["metadata"] == {
        "datafast_visitor_id": "vis_1",
        "datafast_session_id": "ses_1",
    }


@pytest.mark.asyncio
async def test_create_subscription_checkout_attaches_datafast_metadata():
    """create_subscription_checkout threads DataFast IDs into both the session
    metadata and the subscription_data metadata while keeping user_id."""
    mock_session = MagicMock()
    mock_session.url = "https://checkout.stripe.com/pay/cs_test_sub"
    create_mock = MagicMock(return_value=mock_session)
    with (
        patch(
            "backend.data.credit.get_subscription_price_id",
            new_callable=AsyncMock,
            return_value="price_pro_monthly",
        ),
        patch(
            "backend.data.credit.get_stripe_customer_id",
            new_callable=AsyncMock,
            return_value="cus_123",
        ),
        patch(
            "backend.data.credit._expire_open_subscription_sessions",
            new_callable=AsyncMock,
        ),
        patch(
            "backend.data.credit.stripe.checkout.Session.create",
            new=create_mock,
        ),
    ):
        url = await create_subscription_checkout(
            user_id="user-1",
            tier=SubscriptionTier.PRO,
            success_url="https://app.example.com/success",
            cancel_url="https://app.example.com/cancel",
            datafast_visitor_id="vis_1",
            datafast_session_id="ses_1",
        )

    assert url == "https://checkout.stripe.com/pay/cs_test_sub"
    kwargs = create_mock.call_args.kwargs
    assert kwargs["metadata"] == {
        "datafast_visitor_id": "vis_1",
        "datafast_session_id": "ses_1",
    }
    sub_metadata = kwargs["subscription_data"]["metadata"]
    assert sub_metadata["user_id"] == "user-1"
    assert sub_metadata["datafast_visitor_id"] == "vis_1"
    assert sub_metadata["datafast_session_id"] == "ses_1"
