"""
Concurrency and atomicity tests for the credit system.

These tests ensure the credit system handles high-concurrency scenarios correctly
without race conditions, deadlocks, or inconsistent state.
"""

import asyncio
import random
from datetime import datetime

import prisma.enums
import pytest
from prisma.errors import RawQueryError
from prisma.models import CreditTransaction, User

from backend.data.credit import POSTGRES_INT_MAX, UsageTransactionMetadata, UserCredit
from backend.util.exceptions import InsufficientBalanceError
from backend.util.test import SpinTestServer

# Test with both UserCredit and BetaUserCredit if needed
credit_system = UserCredit()


async def create_test_user(user_id: str) -> None:
    """Create a test user with initial balance."""
    try:
        await User.prisma().create(
            data={
                "id": user_id,
                "email": f"test-{user_id}@example.com",
                "name": f"Test User {user_id[:8]}",
                "balance": 0,
            }
        )
    except Exception:
        # User might already exist
        await User.prisma().update(where={"id": user_id}, data={"balance": 0})


async def cleanup_test_user(user_id: str) -> None:
    """Clean up test user and their transactions."""
    try:
        await CreditTransaction.prisma().delete_many(where={"userId": user_id})
        await User.prisma().delete(where={"id": user_id})
    except Exception:
        pass  # Ignore cleanup errors


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_spends_same_user(server: SpinTestServer):
    """Test multiple concurrent spends from the same user don't cause race conditions."""
    user_id = f"concurrent-test-{datetime.now().timestamp()}"
    await create_test_user(user_id)

    try:
        # Give user initial balance
        await credit_system.top_up_credits(user_id, 1000)  # $10

        # Try to spend 10 x $1 concurrently
        async def spend_one_dollar(idx: int):
            try:
                return await credit_system.spend_credits(
                    user_id,
                    100,  # $1
                    UsageTransactionMetadata(
                        graph_exec_id=f"concurrent-{idx}",
                        reason=f"Concurrent spend {idx}",
                    ),
                )
            except InsufficientBalanceError:
                return None

        # Run 10 concurrent spends
        results = await asyncio.gather(
            *[spend_one_dollar(i) for i in range(10)], return_exceptions=True
        )

        # Count successful spends
        successful = [
            r for r in results if r is not None and not isinstance(r, Exception)
        ]
        failed = [r for r in results if isinstance(r, InsufficientBalanceError)]

        # All 10 should succeed since we have exactly $10
        assert len(successful) == 10, f"Expected 10 successful, got {len(successful)}"
        assert len(failed) == 0, f"Expected 0 failures, got {len(failed)}"

        # Final balance should be exactly 0
        final_balance = await credit_system.get_credits(user_id)
        assert final_balance == 0, f"Expected balance 0, got {final_balance}"

        # Verify transaction history is consistent
        transactions = await CreditTransaction.prisma().find_many(
            where={"userId": user_id, "type": prisma.enums.CreditTransactionType.USAGE}
        )
        assert (
            len(transactions) == 10
        ), f"Expected 10 transactions, got {len(transactions)}"

    finally:
        await cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_spends_insufficient_balance(server: SpinTestServer):
    """Test that concurrent spends correctly enforce balance limits."""
    user_id = f"insufficient-test-{datetime.now().timestamp()}"
    await create_test_user(user_id)

    try:
        # Give user limited balance
        await credit_system.top_up_credits(user_id, 500)  # $5

        # Try to spend 10 x $1 concurrently (but only have $5)
        async def spend_one_dollar(idx: int):
            try:
                return await credit_system.spend_credits(
                    user_id,
                    100,  # $1
                    UsageTransactionMetadata(
                        graph_exec_id=f"insufficient-{idx}",
                        reason=f"Insufficient spend {idx}",
                    ),
                )
            except InsufficientBalanceError:
                return "FAILED"

        # Run 10 concurrent spends
        results = await asyncio.gather(
            *[spend_one_dollar(i) for i in range(10)], return_exceptions=True
        )

        # Count successful vs failed
        successful = [
            r
            for r in results
            if r not in ["FAILED", None] and not isinstance(r, Exception)
        ]
        failed = [r for r in results if r == "FAILED"]

        # Exactly 5 should succeed, 5 should fail
        assert len(successful) == 5, f"Expected 5 successful, got {len(successful)}"
        assert len(failed) == 5, f"Expected 5 failures, got {len(failed)}"

        # Final balance should be exactly 0
        final_balance = await credit_system.get_credits(user_id)
        assert final_balance == 0, f"Expected balance 0, got {final_balance}"

    finally:
        await cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_mixed_operations(server: SpinTestServer):
    """Test concurrent mix of spends, top-ups, and balance checks."""
    user_id = f"mixed-test-{datetime.now().timestamp()}"
    await create_test_user(user_id)

    try:
        # Initial balance
        await credit_system.top_up_credits(user_id, 1000)  # $10

        # Mix of operations
        async def mixed_operations():
            operations = []

            # 5 spends of $1 each
            for i in range(5):
                operations.append(
                    credit_system.spend_credits(
                        user_id,
                        100,
                        UsageTransactionMetadata(reason=f"Mixed spend {i}"),
                    )
                )

            # 3 top-ups of $2 each
            for i in range(3):
                operations.append(credit_system.top_up_credits(user_id, 200))

            # 10 balance checks
            for i in range(10):
                operations.append(credit_system.get_credits(user_id))

            return await asyncio.gather(*operations, return_exceptions=True)

        results = await mixed_operations()

        # Check no exceptions occurred
        exceptions = [
            r
            for r in results
            if isinstance(r, Exception) and not isinstance(r, InsufficientBalanceError)
        ]
        assert len(exceptions) == 0, f"Unexpected exceptions: {exceptions}"

        # Final balance should be: 1000 - 500 + 600 = 1100
        final_balance = await credit_system.get_credits(user_id)
        assert final_balance == 1100, f"Expected balance 1100, got {final_balance}"

    finally:
        await cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_race_condition_exact_balance(server: SpinTestServer):
    """Test spending exact balance amount concurrently doesn't go negative."""
    user_id = f"exact-balance-{datetime.now().timestamp()}"
    await create_test_user(user_id)

    try:
        # Give exact amount
        await credit_system.top_up_credits(user_id, 100)  # $1

        # Try to spend $1 twice concurrently
        async def spend_exact():
            try:
                return await credit_system.spend_credits(
                    user_id, 100, UsageTransactionMetadata(reason="Exact spend")
                )
            except InsufficientBalanceError:
                return "FAILED"

        # Both try to spend the full balance
        result1, result2 = await asyncio.gather(spend_exact(), spend_exact())

        # Exactly one should succeed
        results = [result1, result2]
        successful = [
            r for r in results if r != "FAILED" and not isinstance(r, Exception)
        ]
        failed = [r for r in results if r == "FAILED"]

        assert len(successful) == 1, f"Expected 1 success, got {len(successful)}"
        assert len(failed) == 1, f"Expected 1 failure, got {len(failed)}"

        # Balance should be exactly 0, never negative
        final_balance = await credit_system.get_credits(user_id)
        assert final_balance == 0, f"Expected balance 0, got {final_balance}"

    finally:
        await cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_onboarding_reward_idempotency(server: SpinTestServer):
    """Test that onboarding rewards are idempotent (can't be claimed twice)."""
    user_id = f"onboarding-test-{datetime.now().timestamp()}"
    await create_test_user(user_id)

    try:
        # Use WELCOME step which is defined in the OnboardingStep enum
        # Try to claim same reward multiple times concurrently
        async def claim_reward():
            try:
                result = await credit_system.onboarding_reward(
                    user_id, 500, prisma.enums.OnboardingStep.WELCOME
                )
                return "SUCCESS" if result else "DUPLICATE"
            except Exception as e:
                print(f"Claim reward failed: {e}")
                return "FAILED"

        # Try 5 concurrent claims of the same reward
        results = await asyncio.gather(*[claim_reward() for _ in range(5)])

        # Count results
        success_count = results.count("SUCCESS")
        failed_count = results.count("FAILED")

        # At least one should succeed, others should be duplicates
        assert success_count >= 1, f"At least one claim should succeed, got {results}"
        assert failed_count == 0, f"No claims should fail, got {results}"

        # Check balance - should only have 500, not 2500
        final_balance = await credit_system.get_credits(user_id)
        assert final_balance == 500, f"Expected balance 500, got {final_balance}"

        # Check only one transaction exists
        transactions = await CreditTransaction.prisma().find_many(
            where={
                "userId": user_id,
                "type": prisma.enums.CreditTransactionType.GRANT,
                "transactionKey": f"REWARD-{user_id}-WELCOME",
            }
        )
        assert (
            len(transactions) == 1
        ), f"Expected 1 reward transaction, got {len(transactions)}"

    finally:
        await cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_integer_overflow_protection(server: SpinTestServer):
    """Test that integer overflow is prevented."""
    user_id = f"overflow-test-{datetime.now().timestamp()}"
    await create_test_user(user_id)

    try:
        # Try to add amount that would overflow
        max_int = POSTGRES_INT_MAX

        # First, set balance near max
        await User.prisma().update(
            where={"id": user_id}, data={"balance": max_int - 100}
        )

        # Try to add more than possible - should raise database error
        with pytest.raises(RawQueryError, match="integer out of range"):
            await credit_system.top_up_credits(user_id, 200)

        # Balance should not have overflowed
        final_balance = await credit_system.get_credits(user_id)
        assert final_balance <= max_int, f"Balance overflowed: {final_balance}"
        assert (
            final_balance == max_int - 100
        ), f"Balance should be unchanged: {final_balance}"

    finally:
        await cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_high_concurrency_stress(server: SpinTestServer):
    """Stress test with many concurrent operations."""
    user_id = f"stress-test-{datetime.now().timestamp()}"
    await create_test_user(user_id)

    try:
        # Initial balance
        initial_balance = 10000  # $100
        await credit_system.top_up_credits(user_id, initial_balance)

        # Run many concurrent operations
        async def random_operation(idx: int):
            operation = random.choice(["spend", "check"])

            if operation == "spend":
                amount = random.randint(1, 50)  # $0.01 to $0.50
                try:
                    return (
                        "spend",
                        amount,
                        await credit_system.spend_credits(
                            user_id,
                            amount,
                            UsageTransactionMetadata(reason=f"Stress {idx}"),
                        ),
                    )
                except InsufficientBalanceError:
                    return ("spend_failed", amount, None)
            else:
                balance = await credit_system.get_credits(user_id)
                return ("check", 0, balance)

        # Run 100 concurrent operations
        results = await asyncio.gather(
            *[random_operation(i) for i in range(100)], return_exceptions=True
        )

        # Calculate expected final balance
        total_spent = sum(
            r[1]
            for r in results
            if not isinstance(r, Exception) and isinstance(r, tuple) and r[0] == "spend"
        )
        expected_balance = initial_balance - total_spent

        # Verify final balance
        final_balance = await credit_system.get_credits(user_id)
        assert (
            final_balance == expected_balance
        ), f"Expected {expected_balance}, got {final_balance}"
        assert final_balance >= 0, "Balance went negative!"

    finally:
        await cleanup_test_user(user_id)
