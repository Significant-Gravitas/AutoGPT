"""
Test underflow protection for cumulative refunds and negative transactions.

This test ensures that when multiple large refunds are processed, the user balance
doesn't underflow below POSTGRES_INT_MIN, which could cause integer wraparound issues.
"""

import asyncio
from uuid import uuid4

import pytest
from prisma.enums import CreditTransactionType
from prisma.errors import UniqueViolationError
from prisma.models import CreditTransaction, User, UserBalance

from backend.data.credit import POSTGRES_INT_MIN, UserCredit
from backend.util.test import SpinTestServer


async def create_test_user(user_id: str) -> None:
    """Create a test user for underflow tests."""
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
        await UserBalance.prisma().delete_many(where={"userId": user_id})
        await User.prisma().delete_many(where={"id": user_id})
    except Exception as e:
        # Log cleanup failures but don't fail the test
        print(f"Warning: Failed to cleanup test user {user_id}: {e}")


@pytest.mark.asyncio(loop_scope="session")
async def test_debug_underflow_step_by_step(server: SpinTestServer):
    """Debug underflow behavior step by step."""
    credit_system = UserCredit()
    user_id = f"debug-underflow-{uuid4()}"
    await create_test_user(user_id)

    try:
        print(f"POSTGRES_INT_MIN: {POSTGRES_INT_MIN}")

        # Test 1: Set up balance close to underflow threshold
        print("\n=== Test 1: Setting up balance close to underflow threshold ===")
        # First, manually set balance to a value very close to POSTGRES_INT_MIN
        # We'll set it to POSTGRES_INT_MIN + 100, then try to subtract 200
        # This should trigger underflow protection: (POSTGRES_INT_MIN + 100) + (-200) = POSTGRES_INT_MIN - 100
        initial_balance_target = POSTGRES_INT_MIN + 100

        # Use direct database update to set the balance close to underflow
        from prisma.models import UserBalance

        await UserBalance.prisma().upsert(
            where={"userId": user_id},
            data={
                "create": {"userId": user_id, "balance": initial_balance_target},
                "update": {"balance": initial_balance_target},
            },
        )

        current_balance = await credit_system.get_credits(user_id)
        print(f"Set balance to: {current_balance}")
        assert current_balance == initial_balance_target

        # Test 2: Apply amount that should cause underflow
        print("\n=== Test 2: Testing underflow protection ===")
        test_amount = (
            -200
        )  # This should cause underflow: (POSTGRES_INT_MIN + 100) + (-200) = POSTGRES_INT_MIN - 100
        expected_without_protection = current_balance + test_amount
        print(f"Current balance: {current_balance}")
        print(f"Test amount: {test_amount}")
        print(f"Without protection would be: {expected_without_protection}")
        print(f"Should be clamped to POSTGRES_INT_MIN: {POSTGRES_INT_MIN}")

        # Apply the amount that should trigger underflow protection
        balance_result, _ = await credit_system._add_transaction(
            user_id=user_id,
            amount=test_amount,
            transaction_type=CreditTransactionType.REFUND,
            fail_insufficient_credits=False,
        )
        print(f"Actual result: {balance_result}")

        # Check if underflow protection worked
        assert (
            balance_result == POSTGRES_INT_MIN
        ), f"Expected underflow protection to clamp balance to {POSTGRES_INT_MIN}, got {balance_result}"

        # Test 3: Edge case - exactly at POSTGRES_INT_MIN
        print("\n=== Test 3: Testing exact POSTGRES_INT_MIN boundary ===")
        # Set balance to exactly POSTGRES_INT_MIN
        await UserBalance.prisma().upsert(
            where={"userId": user_id},
            data={
                "create": {"userId": user_id, "balance": POSTGRES_INT_MIN},
                "update": {"balance": POSTGRES_INT_MIN},
            },
        )

        edge_balance = await credit_system.get_credits(user_id)
        print(f"Balance set to exactly POSTGRES_INT_MIN: {edge_balance}")

        # Try to subtract 1 - should stay at POSTGRES_INT_MIN
        edge_result, _ = await credit_system._add_transaction(
            user_id=user_id,
            amount=-1,
            transaction_type=CreditTransactionType.REFUND,
            fail_insufficient_credits=False,
        )
        print(f"After subtracting 1: {edge_result}")

        assert (
            edge_result == POSTGRES_INT_MIN
        ), f"Expected balance to remain clamped at {POSTGRES_INT_MIN}, got {edge_result}"

    finally:
        await cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_underflow_protection_large_refunds(server: SpinTestServer):
    """Test that large cumulative refunds don't cause integer underflow."""
    credit_system = UserCredit()
    user_id = f"underflow-test-{uuid4()}"
    await create_test_user(user_id)

    try:
        # Set up balance close to underflow threshold to test the protection
        # Set balance to POSTGRES_INT_MIN + 1000, then try to subtract 2000
        # This should trigger underflow protection
        from prisma.models import UserBalance

        test_balance = POSTGRES_INT_MIN + 1000
        await UserBalance.prisma().upsert(
            where={"userId": user_id},
            data={
                "create": {"userId": user_id, "balance": test_balance},
                "update": {"balance": test_balance},
            },
        )

        current_balance = await credit_system.get_credits(user_id)
        assert current_balance == test_balance

        # Try to deduct amount that would cause underflow: test_balance + (-2000) = POSTGRES_INT_MIN - 1000
        underflow_amount = -2000
        expected_without_protection = (
            current_balance + underflow_amount
        )  # Should be POSTGRES_INT_MIN - 1000

        # Use _add_transaction directly with amount that would cause underflow
        final_balance, _ = await credit_system._add_transaction(
            user_id=user_id,
            amount=underflow_amount,
            transaction_type=CreditTransactionType.REFUND,
            fail_insufficient_credits=False,  # Allow going negative for refunds
        )

        # Balance should be clamped to POSTGRES_INT_MIN, not the calculated underflow value
        assert (
            final_balance == POSTGRES_INT_MIN
        ), f"Balance should be clamped to {POSTGRES_INT_MIN}, got {final_balance}"
        assert (
            final_balance > expected_without_protection
        ), f"Balance should be greater than underflow result {expected_without_protection}, got {final_balance}"

        # Verify with get_credits too
        stored_balance = await credit_system.get_credits(user_id)
        assert (
            stored_balance == POSTGRES_INT_MIN
        ), f"Stored balance should be {POSTGRES_INT_MIN}, got {stored_balance}"

        # Verify transaction was created with the underflow-protected balance
        transactions = await CreditTransaction.prisma().find_many(
            where={"userId": user_id, "type": CreditTransactionType.REFUND},
            order={"createdAt": "desc"},
        )
        assert len(transactions) > 0, "Refund transaction should be created"
        assert (
            transactions[0].runningBalance == POSTGRES_INT_MIN
        ), f"Transaction should show clamped balance {POSTGRES_INT_MIN}, got {transactions[0].runningBalance}"

    finally:
        await cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_multiple_large_refunds_cumulative_underflow(server: SpinTestServer):
    """Test that multiple large refunds applied sequentially don't cause underflow."""
    credit_system = UserCredit()
    user_id = f"cumulative-underflow-test-{uuid4()}"
    await create_test_user(user_id)

    try:
        # Set up balance close to underflow threshold
        from prisma.models import UserBalance

        initial_balance = POSTGRES_INT_MIN + 500  # Close to minimum but with some room
        await UserBalance.prisma().upsert(
            where={"userId": user_id},
            data={
                "create": {"userId": user_id, "balance": initial_balance},
                "update": {"balance": initial_balance},
            },
        )

        # Apply multiple refunds that would cumulatively underflow
        refund_amount = -300  # Each refund that would cause underflow when cumulative

        # First refund: (POSTGRES_INT_MIN + 500) + (-300) = POSTGRES_INT_MIN + 200 (still above minimum)
        balance_1, _ = await credit_system._add_transaction(
            user_id=user_id,
            amount=refund_amount,
            transaction_type=CreditTransactionType.REFUND,
            fail_insufficient_credits=False,
        )

        # Should be above minimum for first refund
        expected_balance_1 = (
            initial_balance + refund_amount
        )  # Should be POSTGRES_INT_MIN + 200
        assert (
            balance_1 == expected_balance_1
        ), f"First refund should result in {expected_balance_1}, got {balance_1}"
        assert (
            balance_1 >= POSTGRES_INT_MIN
        ), f"First refund should not go below {POSTGRES_INT_MIN}, got {balance_1}"

        # Second refund: (POSTGRES_INT_MIN + 200) + (-300) = POSTGRES_INT_MIN - 100 (would underflow)
        balance_2, _ = await credit_system._add_transaction(
            user_id=user_id,
            amount=refund_amount,
            transaction_type=CreditTransactionType.REFUND,
            fail_insufficient_credits=False,
        )

        # Should be clamped to minimum due to underflow protection
        assert (
            balance_2 == POSTGRES_INT_MIN
        ), f"Second refund should be clamped to {POSTGRES_INT_MIN}, got {balance_2}"

        # Third refund: Should stay at minimum
        balance_3, _ = await credit_system._add_transaction(
            user_id=user_id,
            amount=refund_amount,
            transaction_type=CreditTransactionType.REFUND,
            fail_insufficient_credits=False,
        )

        # Should still be at minimum
        assert (
            balance_3 == POSTGRES_INT_MIN
        ), f"Third refund should stay at {POSTGRES_INT_MIN}, got {balance_3}"

        # Final balance check
        final_balance = await credit_system.get_credits(user_id)
        assert (
            final_balance == POSTGRES_INT_MIN
        ), f"Final balance should be {POSTGRES_INT_MIN}, got {final_balance}"

    finally:
        await cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_large_refunds_no_underflow(server: SpinTestServer):
    """Test that concurrent large refunds don't cause race condition underflow."""
    credit_system = UserCredit()
    user_id = f"concurrent-underflow-test-{uuid4()}"
    await create_test_user(user_id)

    try:
        # Set up balance close to underflow threshold
        from prisma.models import UserBalance

        initial_balance = POSTGRES_INT_MIN + 1000  # Close to minimum
        await UserBalance.prisma().upsert(
            where={"userId": user_id},
            data={
                "create": {"userId": user_id, "balance": initial_balance},
                "update": {"balance": initial_balance},
            },
        )

        async def large_refund(amount: int, label: str):
            try:
                return await credit_system._add_transaction(
                    user_id=user_id,
                    amount=-amount,
                    transaction_type=CreditTransactionType.REFUND,
                    fail_insufficient_credits=False,
                )
            except Exception as e:
                return f"FAILED-{label}: {e}"

        # Run concurrent refunds that would cause underflow if not protected
        # Each refund of 500 would cause underflow: initial_balance + (-500) could go below POSTGRES_INT_MIN
        refund_amount = 500
        results = await asyncio.gather(
            large_refund(refund_amount, "A"),
            large_refund(refund_amount, "B"),
            large_refund(refund_amount, "C"),
            return_exceptions=True,
        )

        # Check all results are valid and no underflow occurred
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, tuple):
                balance, _ = result
                assert (
                    balance >= POSTGRES_INT_MIN
                ), f"Result {i} balance {balance} underflowed below {POSTGRES_INT_MIN}"
                valid_results.append(balance)
            elif isinstance(result, str) and "FAILED" in result:
                # Some operations might fail due to validation, that's okay
                pass
            else:
                # Unexpected exception
                assert not isinstance(
                    result, Exception
                ), f"Unexpected exception in result {i}: {result}"

        # At least one operation should succeed
        assert (
            len(valid_results) > 0
        ), f"At least one refund should succeed, got results: {results}"

        # All successful results should be >= POSTGRES_INT_MIN
        for balance in valid_results:
            assert (
                balance >= POSTGRES_INT_MIN
            ), f"Balance {balance} should not be below {POSTGRES_INT_MIN}"

        # Final balance should be valid and at or above POSTGRES_INT_MIN
        final_balance = await credit_system.get_credits(user_id)
        assert (
            final_balance >= POSTGRES_INT_MIN
        ), f"Final balance {final_balance} should not underflow below {POSTGRES_INT_MIN}"

    finally:
        await cleanup_test_user(user_id)
