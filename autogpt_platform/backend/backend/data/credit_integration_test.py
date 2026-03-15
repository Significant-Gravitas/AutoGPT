"""
Integration tests for credit system to catch SQL enum casting issues.

These tests run actual database operations to ensure SQL queries work correctly,
which would have caught the CreditTransactionType enum casting bug.
"""

import pytest
from prisma.enums import CreditTransactionType
from prisma.models import CreditTransaction, User, UserBalance

from backend.data.credit import (
    AutoTopUpConfig,
    BetaUserCredit,
    UsageTransactionMetadata,
    get_auto_top_up,
    set_auto_top_up,
)
from backend.util.json import SafeJson


@pytest.fixture
async def cleanup_test_user():
    """Clean up test user data before and after tests."""
    import uuid

    user_id = str(uuid.uuid4())  # Use unique user ID for each test

    # Create the user first
    try:
        await User.prisma().create(
            data={
                "id": user_id,
                "email": f"test-{user_id}@example.com",
                "topUpConfig": SafeJson({}),
                "timezone": "UTC",
            }
        )
    except Exception:
        # User might already exist, that's fine
        pass

    yield user_id

    # Cleanup after test
    await CreditTransaction.prisma().delete_many(where={"userId": user_id})
    await UserBalance.prisma().delete_many(where={"userId": user_id})
    # Clear auto-top-up config before deleting user
    await User.prisma().update(
        where={"id": user_id}, data={"topUpConfig": SafeJson({})}
    )
    await User.prisma().delete(where={"id": user_id})


@pytest.mark.asyncio(loop_scope="session")
async def test_credit_transaction_enum_casting_integration(cleanup_test_user):
    """
    Integration test to verify CreditTransactionType enum casting works in SQL queries.

    This test would have caught the enum casting bug where PostgreSQL expected
    platform."CreditTransactionType" but got "CreditTransactionType".
    """
    user_id = cleanup_test_user
    credit_system = BetaUserCredit(1000)

    # Test each transaction type to ensure enum casting works
    test_cases = [
        (CreditTransactionType.TOP_UP, 100, "Test top-up"),
        (CreditTransactionType.USAGE, -50, "Test usage"),
        (CreditTransactionType.GRANT, 200, "Test grant"),
        (CreditTransactionType.REFUND, -25, "Test refund"),
        (CreditTransactionType.CARD_CHECK, 0, "Test card check"),
    ]

    for transaction_type, amount, reason in test_cases:
        metadata = SafeJson({"reason": reason, "test": "enum_casting"})

        # This call would fail with enum casting error before the fix
        balance, tx_key = await credit_system._add_transaction(
            user_id=user_id,
            amount=amount,
            transaction_type=transaction_type,
            metadata=metadata,
            is_active=True,
        )

        # Verify transaction was created with correct type
        transaction = await CreditTransaction.prisma().find_first(
            where={"userId": user_id, "transactionKey": tx_key}
        )

        assert transaction is not None
        assert transaction.type == transaction_type
        assert transaction.amount == amount
        assert transaction.metadata is not None

        # Verify metadata content
        assert transaction.metadata["reason"] == reason
        assert transaction.metadata["test"] == "enum_casting"


@pytest.mark.asyncio(loop_scope="session")
async def test_auto_top_up_integration(cleanup_test_user, monkeypatch):
    """
    Integration test for auto-top-up functionality that triggers enum casting.

    This tests the complete auto-top-up flow which involves SQL queries with
    CreditTransactionType enums, ensuring enum casting works end-to-end.
    """
    # Enable credits for this test
    from backend.data.credit import settings

    monkeypatch.setattr(settings.config, "enable_credit", True)
    monkeypatch.setattr(settings.config, "enable_beta_monthly_credit", True)
    monkeypatch.setattr(settings.config, "num_user_credits_refill", 1000)

    user_id = cleanup_test_user
    credit_system = BetaUserCredit(1000)

    # First add some initial credits so we can test the configuration and subsequent behavior
    balance, _ = await credit_system._add_transaction(
        user_id=user_id,
        amount=50,  # Below threshold that we'll set
        transaction_type=CreditTransactionType.GRANT,
        metadata=SafeJson({"reason": "Initial credits before auto top-up config"}),
    )
    assert balance == 50

    # Configure auto top-up with threshold above current balance
    config = AutoTopUpConfig(threshold=100, amount=500)
    await set_auto_top_up(user_id, config)

    # Verify configuration was saved but no immediate top-up occurred
    current_balance = await credit_system.get_credits(user_id)
    assert current_balance == 50  # Balance should be unchanged

    # Simulate spending credits that would trigger auto top-up
    # This involves multiple SQL operations with enum casting
    try:
        metadata = UsageTransactionMetadata(reason="Test spend to trigger auto top-up")
        await credit_system.spend_credits(user_id=user_id, cost=10, metadata=metadata)

        # The auto top-up mechanism should have been triggered
        # Verify the transaction types were handled correctly
        transactions = await CreditTransaction.prisma().find_many(
            where={"userId": user_id}, order={"createdAt": "desc"}
        )

        # Should have at least: GRANT (initial), USAGE (spend), and TOP_UP (auto top-up)
        assert len(transactions) >= 3

        # Verify different transaction types exist and enum casting worked
        transaction_types = {t.type for t in transactions}
        assert CreditTransactionType.GRANT in transaction_types
        assert CreditTransactionType.USAGE in transaction_types
        assert (
            CreditTransactionType.TOP_UP in transaction_types
        )  # Auto top-up should have triggered

    except Exception as e:
        # If this fails with enum casting error, the test successfully caught the bug
        if "CreditTransactionType" in str(e) and (
            "cast" in str(e).lower() or "type" in str(e).lower()
        ):
            pytest.fail(f"Enum casting error detected: {e}")
        else:
            # Re-raise other unexpected errors
            raise


@pytest.mark.asyncio(loop_scope="session")
async def test_enable_transaction_enum_casting_integration(cleanup_test_user):
    """
    Integration test for _enable_transaction with enum casting.

    Tests the scenario where inactive transactions are enabled, which also
    involves SQL queries with CreditTransactionType enum casting.
    """
    user_id = cleanup_test_user
    credit_system = BetaUserCredit(1000)

    # Create an inactive transaction
    balance, tx_key = await credit_system._add_transaction(
        user_id=user_id,
        amount=100,
        transaction_type=CreditTransactionType.TOP_UP,
        metadata=SafeJson({"reason": "Inactive transaction test"}),
        is_active=False,  # Create as inactive
    )

    # Balance should be 0 since transaction is inactive
    assert balance == 0

    # Enable the transaction with new metadata
    enable_metadata = SafeJson(
        {
            "payment_method": "test_payment",
            "activation_reason": "Integration test activation",
        }
    )

    # This would fail with enum casting error before the fix
    final_balance = await credit_system._enable_transaction(
        transaction_key=tx_key,
        user_id=user_id,
        metadata=enable_metadata,
    )

    # Now balance should reflect the activated transaction
    assert final_balance == 100

    # Verify transaction was properly enabled with correct enum type
    transaction = await CreditTransaction.prisma().find_first(
        where={"userId": user_id, "transactionKey": tx_key}
    )

    assert transaction is not None
    assert transaction.isActive is True
    assert transaction.type == CreditTransactionType.TOP_UP
    assert transaction.runningBalance == 100

    # Verify metadata was updated
    assert transaction.metadata is not None
    assert transaction.metadata["payment_method"] == "test_payment"
    assert transaction.metadata["activation_reason"] == "Integration test activation"


@pytest.mark.asyncio(loop_scope="session")
async def test_auto_top_up_configuration_storage(cleanup_test_user, monkeypatch):
    """
    Test that auto-top-up configuration is properly stored and retrieved.

    The immediate top-up logic is handled by the API routes, not the core
    set_auto_top_up function. This test verifies the configuration is correctly
    saved and can be retrieved.
    """
    # Enable credits for this test
    from backend.data.credit import settings

    monkeypatch.setattr(settings.config, "enable_credit", True)
    monkeypatch.setattr(settings.config, "enable_beta_monthly_credit", True)
    monkeypatch.setattr(settings.config, "num_user_credits_refill", 1000)

    user_id = cleanup_test_user
    credit_system = BetaUserCredit(1000)

    # Set initial balance
    balance, _ = await credit_system._add_transaction(
        user_id=user_id,
        amount=50,
        transaction_type=CreditTransactionType.GRANT,
        metadata=SafeJson({"reason": "Initial balance for config test"}),
    )

    assert balance == 50

    # Configure auto top-up
    config = AutoTopUpConfig(threshold=100, amount=200)
    await set_auto_top_up(user_id, config)

    # Verify the configuration was saved
    retrieved_config = await get_auto_top_up(user_id)
    assert retrieved_config.threshold == config.threshold
    assert retrieved_config.amount == config.amount

    # Verify balance is unchanged (no immediate top-up from set_auto_top_up)
    final_balance = await credit_system.get_credits(user_id)
    assert final_balance == 50  # Should be unchanged

    # Verify no immediate auto-top-up transaction was created by set_auto_top_up
    transactions = await CreditTransaction.prisma().find_many(
        where={"userId": user_id}, order={"createdAt": "desc"}
    )

    # Should only have the initial GRANT transaction
    assert len(transactions) == 1
    assert transactions[0].type == CreditTransactionType.GRANT
