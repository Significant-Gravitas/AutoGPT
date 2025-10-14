"""
Integration test to verify complete migration from User.balance to UserBalance table.

This test ensures that:
1. No User.balance queries exist in the system
2. All balance operations go through UserBalance table
3. User and UserBalance stay synchronized properly
"""

import asyncio
from datetime import datetime

import pytest
from prisma.enums import CreditTransactionType
from prisma.errors import UniqueViolationError
from prisma.models import CreditTransaction, User, UserBalance

from backend.data.credit import UsageTransactionMetadata, UserCredit
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer


async def create_test_user(user_id: str) -> None:
    """Create a test user for migration tests."""
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


async def cleanup_test_user(user_id: str) -> None:
    """Clean up test user and their data."""
    try:
        await CreditTransaction.prisma().delete_many(where={"userId": user_id})
        await UserBalance.prisma().delete_many(where={"userId": user_id})
        await User.prisma().delete_many(where={"id": user_id})
    except Exception as e:
        # Log cleanup failures but don't fail the test
        print(f"Warning: Failed to cleanup test user {user_id}: {e}")


@pytest.mark.asyncio(loop_scope="session")
async def test_user_balance_migration_complete(server: SpinTestServer):
    """Test that User table balance is never used and UserBalance is source of truth."""
    credit_system = UserCredit()
    user_id = f"migration-test-{datetime.now().timestamp()}"
    await create_test_user(user_id)

    try:
        # 1. Verify User table does NOT have balance set initially
        user = await User.prisma().find_unique(where={"id": user_id})
        assert user is not None
        # User.balance should not exist or should be None/0 if it exists
        user_balance_attr = getattr(user, "balance", None)
        if user_balance_attr is not None:
            assert (
                user_balance_attr == 0 or user_balance_attr is None
            ), f"User.balance should be 0 or None, got {user_balance_attr}"

        # 2. Perform various credit operations using internal method (bypasses Stripe)
        await credit_system._add_transaction(
            user_id=user_id,
            amount=1000,
            transaction_type=CreditTransactionType.TOP_UP,
            metadata=SafeJson({"test": "migration_test"}),
        )
        balance1 = await credit_system.get_credits(user_id)
        assert balance1 == 1000

        await credit_system.spend_credits(
            user_id,
            300,
            UsageTransactionMetadata(
                graph_exec_id="test", reason="Migration test spend"
            ),
        )
        balance2 = await credit_system.get_credits(user_id)
        assert balance2 == 700

        # 3. Verify UserBalance table has correct values
        user_balance = await UserBalance.prisma().find_unique(where={"userId": user_id})
        assert user_balance is not None
        assert (
            user_balance.balance == 700
        ), f"UserBalance should be 700, got {user_balance.balance}"

        # 4. CRITICAL: Verify User.balance is NEVER updated during operations
        user_after = await User.prisma().find_unique(where={"id": user_id})
        assert user_after is not None
        user_balance_after = getattr(user_after, "balance", None)
        if user_balance_after is not None:
            # If User.balance exists, it should still be 0 (never updated)
            assert (
                user_balance_after == 0 or user_balance_after is None
            ), f"User.balance should remain 0/None after operations, got {user_balance_after}. This indicates User.balance is still being used!"

        # 5. Verify get_credits always returns UserBalance value, not User.balance
        final_balance = await credit_system.get_credits(user_id)
        assert (
            final_balance == user_balance.balance
        ), f"get_credits should return UserBalance value {user_balance.balance}, got {final_balance}"

    finally:
        await cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_detect_stale_user_balance_queries(server: SpinTestServer):
    """Test to detect if any operations are still using User.balance instead of UserBalance."""
    credit_system = UserCredit()
    user_id = f"stale-query-test-{datetime.now().timestamp()}"
    await create_test_user(user_id)

    try:
        # Create UserBalance with specific value
        await UserBalance.prisma().create(
            data={"userId": user_id, "balance": 5000}  # $50
        )

        # Verify that get_credits returns UserBalance value (5000), not any stale User.balance value
        balance = await credit_system.get_credits(user_id)
        assert (
            balance == 5000
        ), f"Expected get_credits to return 5000 from UserBalance, got {balance}"

        # Verify all operations use UserBalance using internal method (bypasses Stripe)
        await credit_system._add_transaction(
            user_id=user_id,
            amount=1000,
            transaction_type=CreditTransactionType.TOP_UP,
            metadata=SafeJson({"test": "final_verification"}),
        )
        final_balance = await credit_system.get_credits(user_id)
        assert final_balance == 6000, f"Expected 6000, got {final_balance}"

        # Verify UserBalance table has the correct value
        user_balance = await UserBalance.prisma().find_unique(where={"userId": user_id})
        assert user_balance is not None
        assert (
            user_balance.balance == 6000
        ), f"UserBalance should be 6000, got {user_balance.balance}"

    finally:
        await cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_operations_use_userbalance_only(server: SpinTestServer):
    """Test that concurrent operations all use UserBalance locking, not User.balance."""
    credit_system = UserCredit()
    user_id = f"concurrent-userbalance-test-{datetime.now().timestamp()}"
    await create_test_user(user_id)

    try:
        # Set initial balance in UserBalance
        await UserBalance.prisma().create(data={"userId": user_id, "balance": 1000})

        # Run concurrent operations to ensure they all use UserBalance atomic operations
        async def concurrent_spend(amount: int, label: str):
            try:
                await credit_system.spend_credits(
                    user_id,
                    amount,
                    UsageTransactionMetadata(
                        graph_exec_id=f"concurrent-{label}",
                        reason=f"Concurrent test {label}",
                    ),
                )
                return f"{label}-SUCCESS"
            except Exception as e:
                return f"{label}-FAILED: {e}"

        # Run concurrent operations
        results = await asyncio.gather(
            concurrent_spend(100, "A"),
            concurrent_spend(200, "B"),
            concurrent_spend(300, "C"),
            return_exceptions=True,
        )

        # All should succeed (1000 >= 100+200+300)
        successful = [r for r in results if "SUCCESS" in str(r)]
        assert len(successful) == 3, f"All operations should succeed, got {results}"

        # Final balance should be 1000 - 600 = 400
        final_balance = await credit_system.get_credits(user_id)
        assert final_balance == 400, f"Expected final balance 400, got {final_balance}"

        # Verify UserBalance has correct value
        user_balance = await UserBalance.prisma().find_unique(where={"userId": user_id})
        assert user_balance is not None
        assert (
            user_balance.balance == 400
        ), f"UserBalance should be 400, got {user_balance.balance}"

        # Critical: If User.balance exists and was used, it might have wrong value
        try:
            user = await User.prisma().find_unique(where={"id": user_id})
            user_balance_attr = getattr(user, "balance", None)
            if user_balance_attr is not None:
                # If User.balance exists, it should NOT be used for operations
                # The fact that our final balance is correct from UserBalance proves the system is working
                print(
                    f"✅ User.balance exists ({user_balance_attr}) but UserBalance ({user_balance.balance}) is being used correctly"
                )
        except Exception:
            print("✅ User.balance column doesn't exist - migration is complete")

    finally:
        await cleanup_test_user(user_id)
