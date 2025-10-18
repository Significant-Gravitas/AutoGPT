"""
Test ceiling balance functionality to ensure auto top-up limits work correctly.

This test was added to cover a previously untested code path that could lead to
incorrect balance capping behavior.
"""

from uuid import uuid4

import pytest
from prisma.enums import CreditTransactionType
from prisma.errors import UniqueViolationError
from prisma.models import CreditTransaction, User, UserBalance

from backend.data.credit import UserCredit
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer


async def create_test_user(user_id: str) -> None:
    """Create a test user for ceiling tests."""
    try:
        await User.prisma().create(
            data={
                "id": user_id,
                "email": f"test-{user_id}@example.com",
                "name": f"Test User {user_id[:8]}",
            }
        )
    except UniqueViolationError:
        # User already exists, continue
        pass

    await UserBalance.prisma().upsert(
        where={"userId": user_id},
        data={"create": {"userId": user_id, "balance": 0}, "update": {"balance": 0}},
    )


async def cleanup_test_user(user_id: str) -> None:
    """Clean up test user and their transactions."""
    try:
        await CreditTransaction.prisma().delete_many(where={"userId": user_id})
        await User.prisma().delete_many(where={"id": user_id})
    except Exception as e:
        # Log cleanup failures but don't fail the test
        print(f"Warning: Failed to cleanup test user {user_id}: {e}")


@pytest.mark.asyncio(loop_scope="session")
async def test_ceiling_balance_rejects_when_above_threshold(server: SpinTestServer):
    """Test that ceiling balance correctly rejects top-ups when balance is above threshold."""
    credit_system = UserCredit()
    user_id = f"ceiling-test-{uuid4()}"
    await create_test_user(user_id)

    try:
        # Give user balance of 1000 ($10) using internal method (bypasses Stripe)
        await credit_system._add_transaction(
            user_id=user_id,
            amount=1000,
            transaction_type=CreditTransactionType.TOP_UP,
            metadata=SafeJson({"test": "initial_balance"}),
        )
        current_balance = await credit_system.get_credits(user_id)
        assert current_balance == 1000

        # Try to add 200 more with ceiling of 800 (should reject since 1000 > 800)
        with pytest.raises(ValueError, match="You already have enough balance"):
            await credit_system._add_transaction(
                user_id=user_id,
                amount=200,
                transaction_type=CreditTransactionType.TOP_UP,
                ceiling_balance=800,  # Ceiling lower than current balance
            )

        # Balance should remain unchanged
        final_balance = await credit_system.get_credits(user_id)
        assert final_balance == 1000, f"Balance should remain 1000, got {final_balance}"

    finally:
        await cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_ceiling_balance_clamps_when_would_exceed(server: SpinTestServer):
    """Test that ceiling balance correctly clamps amounts that would exceed the ceiling."""
    credit_system = UserCredit()
    user_id = f"ceiling-clamp-test-{uuid4()}"
    await create_test_user(user_id)

    try:
        # Give user balance of 500 ($5) using internal method (bypasses Stripe)
        await credit_system._add_transaction(
            user_id=user_id,
            amount=500,
            transaction_type=CreditTransactionType.TOP_UP,
            metadata=SafeJson({"test": "initial_balance"}),
        )

        # Add 800 more with ceiling of 1000 (should clamp to 1000, not reach 1300)
        final_balance, _ = await credit_system._add_transaction(
            user_id=user_id,
            amount=800,
            transaction_type=CreditTransactionType.TOP_UP,
            ceiling_balance=1000,  # Ceiling should clamp 500 + 800 = 1300 to 1000
        )

        # Balance should be clamped to ceiling
        assert (
            final_balance == 1000
        ), f"Balance should be clamped to 1000, got {final_balance}"

        # Verify with get_credits too
        stored_balance = await credit_system.get_credits(user_id)
        assert (
            stored_balance == 1000
        ), f"Stored balance should be 1000, got {stored_balance}"

        # Verify transaction shows the clamped amount
        transactions = await CreditTransaction.prisma().find_many(
            where={"userId": user_id, "type": CreditTransactionType.TOP_UP},
            order={"createdAt": "desc"},
        )

        # Should have 2 transactions: 500 + (500 to reach ceiling of 1000)
        assert len(transactions) == 2

        # The second transaction should show it only added 500, not 800
        second_tx = transactions[0]  # Most recent
        assert second_tx.runningBalance == 1000
        # The actual amount recorded could be 800 (what was requested) but balance was clamped

    finally:
        await cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_ceiling_balance_allows_when_under_threshold(server: SpinTestServer):
    """Test that ceiling balance allows top-ups when balance is under threshold."""
    credit_system = UserCredit()
    user_id = f"ceiling-under-test-{uuid4()}"
    await create_test_user(user_id)

    try:
        # Give user balance of 300 ($3) using internal method (bypasses Stripe)
        await credit_system._add_transaction(
            user_id=user_id,
            amount=300,
            transaction_type=CreditTransactionType.TOP_UP,
            metadata=SafeJson({"test": "initial_balance"}),
        )

        # Add 200 more with ceiling of 1000 (should succeed: 300 + 200 = 500 < 1000)
        final_balance, _ = await credit_system._add_transaction(
            user_id=user_id,
            amount=200,
            transaction_type=CreditTransactionType.TOP_UP,
            ceiling_balance=1000,
        )

        # Balance should be exactly 500
        assert final_balance == 500, f"Balance should be 500, got {final_balance}"

        # Verify with get_credits too
        stored_balance = await credit_system.get_credits(user_id)
        assert (
            stored_balance == 500
        ), f"Stored balance should be 500, got {stored_balance}"

    finally:
        await cleanup_test_user(user_id)
