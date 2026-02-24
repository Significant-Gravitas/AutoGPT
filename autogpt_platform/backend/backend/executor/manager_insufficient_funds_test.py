from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import NotificationType

from backend.data.notifications import ZeroBalanceData
from backend.executor.manager import (
    INSUFFICIENT_FUNDS_NOTIFIED_PREFIX,
    ExecutionProcessor,
    clear_insufficient_funds_notifications,
)
from backend.util.exceptions import InsufficientBalanceError
from backend.util.test import SpinTestServer


async def async_iter(items):
    """Helper to create an async iterator from a list."""
    for item in items:
        yield item


@pytest.mark.asyncio(loop_scope="session")
async def test_handle_insufficient_funds_sends_discord_alert_first_time(
    server: SpinTestServer,
):
    """Test that the first insufficient funds notification sends a Discord alert."""

    execution_processor = ExecutionProcessor()
    user_id = "test-user-123"
    graph_id = "test-graph-456"
    error = InsufficientBalanceError(
        message="Insufficient balance",
        user_id=user_id,
        balance=72,  # $0.72
        amount=-714,  # Attempting to spend $7.14
    )

    with patch(
        "backend.executor.manager.queue_notification"
    ) as mock_queue_notif, patch(
        "backend.executor.manager.get_notification_manager_client"
    ) as mock_get_client, patch(
        "backend.executor.manager.settings"
    ) as mock_settings, patch(
        "backend.executor.manager.redis"
    ) as mock_redis_module:

        # Setup mocks
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_settings.config.frontend_base_url = "https://test.com"

        # Mock Redis to simulate first-time notification (set returns True)
        mock_redis_client = MagicMock()
        mock_redis_module.get_redis.return_value = mock_redis_client
        mock_redis_client.set.return_value = True  # Key was newly set

        # Create mock database client
        mock_db_client = MagicMock()
        mock_graph_metadata = MagicMock()
        mock_graph_metadata.name = "Test Agent"
        mock_db_client.get_graph_metadata.return_value = mock_graph_metadata
        mock_db_client.get_user_email_by_id.return_value = "test@example.com"

        # Test the insufficient funds handler
        execution_processor._handle_insufficient_funds_notif(
            db_client=mock_db_client,
            user_id=user_id,
            graph_id=graph_id,
            e=error,
        )

        # Verify notification was queued
        mock_queue_notif.assert_called_once()
        notification_call = mock_queue_notif.call_args[0][0]
        assert notification_call.type == NotificationType.ZERO_BALANCE
        assert notification_call.user_id == user_id
        assert isinstance(notification_call.data, ZeroBalanceData)
        assert notification_call.data.current_balance == 72

        # Verify Redis was checked with correct key pattern
        expected_key = f"{INSUFFICIENT_FUNDS_NOTIFIED_PREFIX}:{user_id}:{graph_id}"
        mock_redis_client.set.assert_called_once()
        call_args = mock_redis_client.set.call_args
        assert call_args[0][0] == expected_key
        assert call_args[1]["nx"] is True

        # Verify Discord alert was sent
        mock_client.discord_system_alert.assert_called_once()
        discord_message = mock_client.discord_system_alert.call_args[0][0]
        assert "Insufficient Funds Alert" in discord_message
        assert "test@example.com" in discord_message
        assert "Test Agent" in discord_message


@pytest.mark.asyncio(loop_scope="session")
async def test_handle_insufficient_funds_skips_duplicate_notifications(
    server: SpinTestServer,
):
    """Test that duplicate insufficient funds notifications skip both email and Discord."""

    execution_processor = ExecutionProcessor()
    user_id = "test-user-123"
    graph_id = "test-graph-456"
    error = InsufficientBalanceError(
        message="Insufficient balance",
        user_id=user_id,
        balance=72,
        amount=-714,
    )

    with patch(
        "backend.executor.manager.queue_notification"
    ) as mock_queue_notif, patch(
        "backend.executor.manager.get_notification_manager_client"
    ) as mock_get_client, patch(
        "backend.executor.manager.settings"
    ) as mock_settings, patch(
        "backend.executor.manager.redis"
    ) as mock_redis_module:

        # Setup mocks
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_settings.config.frontend_base_url = "https://test.com"

        # Mock Redis to simulate duplicate notification (set returns False/None)
        mock_redis_client = MagicMock()
        mock_redis_module.get_redis.return_value = mock_redis_client
        mock_redis_client.set.return_value = None  # Key already existed

        # Create mock database client
        mock_db_client = MagicMock()
        mock_db_client.get_graph_metadata.return_value = MagicMock(name="Test Agent")

        # Test the insufficient funds handler
        execution_processor._handle_insufficient_funds_notif(
            db_client=mock_db_client,
            user_id=user_id,
            graph_id=graph_id,
            e=error,
        )

        # Verify email notification was NOT queued (deduplication worked)
        mock_queue_notif.assert_not_called()

        # Verify Discord alert was NOT sent (deduplication worked)
        mock_client.discord_system_alert.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_handle_insufficient_funds_different_agents_get_separate_alerts(
    server: SpinTestServer,
):
    """Test that different agents for the same user get separate Discord alerts."""

    execution_processor = ExecutionProcessor()
    user_id = "test-user-123"
    graph_id_1 = "test-graph-111"
    graph_id_2 = "test-graph-222"

    error = InsufficientBalanceError(
        message="Insufficient balance",
        user_id=user_id,
        balance=72,
        amount=-714,
    )

    with patch("backend.executor.manager.queue_notification"), patch(
        "backend.executor.manager.get_notification_manager_client"
    ) as mock_get_client, patch(
        "backend.executor.manager.settings"
    ) as mock_settings, patch(
        "backend.executor.manager.redis"
    ) as mock_redis_module:

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_settings.config.frontend_base_url = "https://test.com"

        mock_redis_client = MagicMock()
        mock_redis_module.get_redis.return_value = mock_redis_client
        # Both calls return True (first time for each agent)
        mock_redis_client.set.return_value = True

        mock_db_client = MagicMock()
        mock_graph_metadata = MagicMock()
        mock_graph_metadata.name = "Test Agent"
        mock_db_client.get_graph_metadata.return_value = mock_graph_metadata
        mock_db_client.get_user_email_by_id.return_value = "test@example.com"

        # First agent notification
        execution_processor._handle_insufficient_funds_notif(
            db_client=mock_db_client,
            user_id=user_id,
            graph_id=graph_id_1,
            e=error,
        )

        # Second agent notification
        execution_processor._handle_insufficient_funds_notif(
            db_client=mock_db_client,
            user_id=user_id,
            graph_id=graph_id_2,
            e=error,
        )

        # Verify Discord alerts were sent for both agents
        assert mock_client.discord_system_alert.call_count == 2

        # Verify Redis was called with different keys
        assert mock_redis_client.set.call_count == 2
        calls = mock_redis_client.set.call_args_list
        assert (
            calls[0][0][0]
            == f"{INSUFFICIENT_FUNDS_NOTIFIED_PREFIX}:{user_id}:{graph_id_1}"
        )
        assert (
            calls[1][0][0]
            == f"{INSUFFICIENT_FUNDS_NOTIFIED_PREFIX}:{user_id}:{graph_id_2}"
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_clear_insufficient_funds_notifications(server: SpinTestServer):
    """Test that clearing notifications removes all keys for a user."""

    user_id = "test-user-123"

    with patch("backend.executor.manager.redis") as mock_redis_module:

        mock_redis_client = MagicMock()
        # get_redis_async is an async function, so we need AsyncMock for it
        mock_redis_module.get_redis_async = AsyncMock(return_value=mock_redis_client)

        # Mock scan_iter to return some keys as an async iterator
        mock_keys = [
            f"{INSUFFICIENT_FUNDS_NOTIFIED_PREFIX}:{user_id}:graph-1",
            f"{INSUFFICIENT_FUNDS_NOTIFIED_PREFIX}:{user_id}:graph-2",
            f"{INSUFFICIENT_FUNDS_NOTIFIED_PREFIX}:{user_id}:graph-3",
        ]
        mock_redis_client.scan_iter.return_value = async_iter(mock_keys)
        # delete is awaited, so use AsyncMock
        mock_redis_client.delete = AsyncMock(return_value=3)

        # Clear notifications
        result = await clear_insufficient_funds_notifications(user_id)

        # Verify correct pattern was used
        expected_pattern = f"{INSUFFICIENT_FUNDS_NOTIFIED_PREFIX}:{user_id}:*"
        mock_redis_client.scan_iter.assert_called_once_with(match=expected_pattern)

        # Verify delete was called with all keys
        mock_redis_client.delete.assert_called_once_with(*mock_keys)

        # Verify return value
        assert result == 3


@pytest.mark.asyncio(loop_scope="session")
async def test_clear_insufficient_funds_notifications_no_keys(server: SpinTestServer):
    """Test clearing notifications when there are no keys to clear."""

    user_id = "test-user-no-notifications"

    with patch("backend.executor.manager.redis") as mock_redis_module:

        mock_redis_client = MagicMock()
        # get_redis_async is an async function, so we need AsyncMock for it
        mock_redis_module.get_redis_async = AsyncMock(return_value=mock_redis_client)

        # Mock scan_iter to return no keys as an async iterator
        mock_redis_client.scan_iter.return_value = async_iter([])

        # Clear notifications
        result = await clear_insufficient_funds_notifications(user_id)

        # Verify delete was not called
        mock_redis_client.delete.assert_not_called()

        # Verify return value
        assert result == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_clear_insufficient_funds_notifications_handles_redis_error(
    server: SpinTestServer,
):
    """Test that clearing notifications handles Redis errors gracefully."""

    user_id = "test-user-redis-error"

    with patch("backend.executor.manager.redis") as mock_redis_module:

        # Mock get_redis_async to raise an error
        mock_redis_module.get_redis_async = AsyncMock(
            side_effect=Exception("Redis connection failed")
        )

        # Clear notifications should not raise, just return 0
        result = await clear_insufficient_funds_notifications(user_id)

        # Verify it returned 0 (graceful failure)
        assert result == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_handle_insufficient_funds_continues_on_redis_error(
    server: SpinTestServer,
):
    """Test that both email and Discord notifications are still sent when Redis fails."""

    execution_processor = ExecutionProcessor()
    user_id = "test-user-123"
    graph_id = "test-graph-456"
    error = InsufficientBalanceError(
        message="Insufficient balance",
        user_id=user_id,
        balance=72,
        amount=-714,
    )

    with patch(
        "backend.executor.manager.queue_notification"
    ) as mock_queue_notif, patch(
        "backend.executor.manager.get_notification_manager_client"
    ) as mock_get_client, patch(
        "backend.executor.manager.settings"
    ) as mock_settings, patch(
        "backend.executor.manager.redis"
    ) as mock_redis_module:

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_settings.config.frontend_base_url = "https://test.com"

        # Mock Redis to raise an error
        mock_redis_client = MagicMock()
        mock_redis_module.get_redis.return_value = mock_redis_client
        mock_redis_client.set.side_effect = Exception("Redis connection error")

        mock_db_client = MagicMock()
        mock_graph_metadata = MagicMock()
        mock_graph_metadata.name = "Test Agent"
        mock_db_client.get_graph_metadata.return_value = mock_graph_metadata
        mock_db_client.get_user_email_by_id.return_value = "test@example.com"

        # Test the insufficient funds handler
        execution_processor._handle_insufficient_funds_notif(
            db_client=mock_db_client,
            user_id=user_id,
            graph_id=graph_id,
            e=error,
        )

        # Verify email notification was still queued despite Redis error
        mock_queue_notif.assert_called_once()

        # Verify Discord alert was still sent despite Redis error
        mock_client.discord_system_alert.assert_called_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_add_transaction_clears_notifications_on_grant(server: SpinTestServer):
    """Test that _add_transaction clears notification flags when adding GRANT credits."""
    from prisma.enums import CreditTransactionType

    from backend.data.credit import UserCredit

    user_id = "test-user-grant-clear"

    with patch("backend.data.credit.query_raw_with_schema") as mock_query, patch(
        "backend.executor.manager.redis"
    ) as mock_redis_module:

        # Mock the query to return a successful transaction
        mock_query.return_value = [{"balance": 1000, "transactionKey": "test-tx-key"}]

        # Mock async Redis for notification clearing
        mock_redis_client = MagicMock()
        mock_redis_module.get_redis_async = AsyncMock(return_value=mock_redis_client)
        mock_redis_client.scan_iter.return_value = async_iter(
            [f"{INSUFFICIENT_FUNDS_NOTIFIED_PREFIX}:{user_id}:graph-1"]
        )
        mock_redis_client.delete = AsyncMock(return_value=1)

        # Create a concrete instance
        credit_model = UserCredit()

        # Call _add_transaction with GRANT type (should clear notifications)
        await credit_model._add_transaction(
            user_id=user_id,
            amount=500,  # Positive amount
            transaction_type=CreditTransactionType.GRANT,
            is_active=True,  # Active transaction
        )

        # Verify notification clearing was called
        mock_redis_module.get_redis_async.assert_called_once()
        mock_redis_client.scan_iter.assert_called_once_with(
            match=f"{INSUFFICIENT_FUNDS_NOTIFIED_PREFIX}:{user_id}:*"
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_add_transaction_clears_notifications_on_top_up(server: SpinTestServer):
    """Test that _add_transaction clears notification flags when adding TOP_UP credits."""
    from prisma.enums import CreditTransactionType

    from backend.data.credit import UserCredit

    user_id = "test-user-topup-clear"

    with patch("backend.data.credit.query_raw_with_schema") as mock_query, patch(
        "backend.executor.manager.redis"
    ) as mock_redis_module:

        # Mock the query to return a successful transaction
        mock_query.return_value = [{"balance": 2000, "transactionKey": "test-tx-key-2"}]

        # Mock async Redis for notification clearing
        mock_redis_client = MagicMock()
        mock_redis_module.get_redis_async = AsyncMock(return_value=mock_redis_client)
        mock_redis_client.scan_iter.return_value = async_iter([])
        mock_redis_client.delete = AsyncMock(return_value=0)

        credit_model = UserCredit()

        # Call _add_transaction with TOP_UP type (should clear notifications)
        await credit_model._add_transaction(
            user_id=user_id,
            amount=1000,  # Positive amount
            transaction_type=CreditTransactionType.TOP_UP,
            is_active=True,
        )

        # Verify notification clearing was attempted
        mock_redis_module.get_redis_async.assert_called_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_add_transaction_skips_clearing_for_inactive_transaction(
    server: SpinTestServer,
):
    """Test that _add_transaction does NOT clear notifications for inactive transactions."""
    from prisma.enums import CreditTransactionType

    from backend.data.credit import UserCredit

    user_id = "test-user-inactive"

    with patch("backend.data.credit.query_raw_with_schema") as mock_query, patch(
        "backend.executor.manager.redis"
    ) as mock_redis_module:

        # Mock the query to return a successful transaction
        mock_query.return_value = [{"balance": 500, "transactionKey": "test-tx-key-3"}]

        # Mock async Redis
        mock_redis_client = MagicMock()
        mock_redis_module.get_redis_async = AsyncMock(return_value=mock_redis_client)

        credit_model = UserCredit()

        # Call _add_transaction with is_active=False (should NOT clear notifications)
        await credit_model._add_transaction(
            user_id=user_id,
            amount=500,
            transaction_type=CreditTransactionType.TOP_UP,
            is_active=False,  # Inactive - pending Stripe payment
        )

        # Verify notification clearing was NOT called
        mock_redis_module.get_redis_async.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_add_transaction_skips_clearing_for_usage_transaction(
    server: SpinTestServer,
):
    """Test that _add_transaction does NOT clear notifications for USAGE transactions."""
    from prisma.enums import CreditTransactionType

    from backend.data.credit import UserCredit

    user_id = "test-user-usage"

    with patch("backend.data.credit.query_raw_with_schema") as mock_query, patch(
        "backend.executor.manager.redis"
    ) as mock_redis_module:

        # Mock the query to return a successful transaction
        mock_query.return_value = [{"balance": 400, "transactionKey": "test-tx-key-4"}]

        # Mock async Redis
        mock_redis_client = MagicMock()
        mock_redis_module.get_redis_async = AsyncMock(return_value=mock_redis_client)

        credit_model = UserCredit()

        # Call _add_transaction with USAGE type (spending, should NOT clear)
        await credit_model._add_transaction(
            user_id=user_id,
            amount=-100,  # Negative - spending credits
            transaction_type=CreditTransactionType.USAGE,
            is_active=True,
        )

        # Verify notification clearing was NOT called
        mock_redis_module.get_redis_async.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_enable_transaction_clears_notifications(server: SpinTestServer):
    """Test that _enable_transaction clears notification flags when enabling a TOP_UP."""
    from prisma.enums import CreditTransactionType

    from backend.data.credit import UserCredit

    user_id = "test-user-enable"

    with patch("backend.data.credit.CreditTransaction") as mock_credit_tx, patch(
        "backend.data.credit.query_raw_with_schema"
    ) as mock_query, patch("backend.executor.manager.redis") as mock_redis_module:

        # Mock finding the pending transaction
        mock_transaction = MagicMock()
        mock_transaction.amount = 1000
        mock_transaction.type = CreditTransactionType.TOP_UP
        mock_credit_tx.prisma.return_value.find_first = AsyncMock(
            return_value=mock_transaction
        )

        # Mock the query to return updated balance
        mock_query.return_value = [{"balance": 1500}]

        # Mock async Redis for notification clearing
        mock_redis_client = MagicMock()
        mock_redis_module.get_redis_async = AsyncMock(return_value=mock_redis_client)
        mock_redis_client.scan_iter.return_value = async_iter(
            [f"{INSUFFICIENT_FUNDS_NOTIFIED_PREFIX}:{user_id}:graph-1"]
        )
        mock_redis_client.delete = AsyncMock(return_value=1)

        credit_model = UserCredit()

        # Call _enable_transaction (simulates Stripe checkout completion)
        from backend.util.json import SafeJson

        await credit_model._enable_transaction(
            transaction_key="cs_test_123",
            user_id=user_id,
            metadata=SafeJson({"payment": "completed"}),
        )

        # Verify notification clearing was called
        mock_redis_module.get_redis_async.assert_called_once()
        mock_redis_client.scan_iter.assert_called_once_with(
            match=f"{INSUFFICIENT_FUNDS_NOTIFIED_PREFIX}:{user_id}:*"
        )
