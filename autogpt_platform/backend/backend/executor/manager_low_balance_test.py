from unittest.mock import MagicMock, patch

import pytest
from prisma.enums import NotificationType

from backend.data.notifications import LowBalanceData
from backend.executor.manager import ExecutionProcessor
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_handle_low_balance_threshold_crossing(server: SpinTestServer):
    """Test that _handle_low_balance triggers notification when crossing threshold."""

    execution_processor = ExecutionProcessor()
    user_id = "test-user-123"
    current_balance = 400  # $4 - below $5 threshold
    transaction_cost = 600  # $6 transaction

    # Mock dependencies
    with patch(
        "backend.executor.manager.queue_notification"
    ) as mock_queue_notif, patch(
        "backend.executor.manager.get_notification_manager_client"
    ) as mock_get_client, patch(
        "backend.executor.manager.settings"
    ) as mock_settings:

        # Setup mocks
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_settings.config.low_balance_threshold = 500  # $5 threshold
        mock_settings.config.frontend_base_url = "https://test.com"

        # Create mock database client
        mock_db_client = MagicMock()
        mock_db_client.get_user_email_by_id.return_value = "test@example.com"

        # Test the low balance handler
        execution_processor._handle_low_balance(
            db_client=mock_db_client,
            user_id=user_id,
            current_balance=current_balance,
            transaction_cost=transaction_cost,
        )

        # Verify notification was queued
        mock_queue_notif.assert_called_once()
        notification_call = mock_queue_notif.call_args[0][0]

        # Verify notification details
        assert notification_call.type == NotificationType.LOW_BALANCE
        assert notification_call.user_id == user_id
        assert isinstance(notification_call.data, LowBalanceData)
        assert notification_call.data.current_balance == current_balance

        # Verify Discord alert was sent
        mock_client.discord_system_alert.assert_called_once()
        discord_message = mock_client.discord_system_alert.call_args[0][0]
        assert "Low Balance Alert" in discord_message
        assert "test@example.com" in discord_message
        assert "$4.00" in discord_message
        assert "$6.00" in discord_message


@pytest.mark.asyncio(loop_scope="session")
async def test_handle_low_balance_no_notification_when_not_crossing(
    server: SpinTestServer,
):
    """Test that no notification is sent when not crossing the threshold."""

    execution_processor = ExecutionProcessor()
    user_id = "test-user-123"
    current_balance = 600  # $6 - above $5 threshold
    transaction_cost = (
        100  # $1 transaction (balance before was $7, still above threshold)
    )

    # Mock dependencies
    with patch(
        "backend.executor.manager.queue_notification"
    ) as mock_queue_notif, patch(
        "backend.executor.manager.get_notification_manager_client"
    ) as mock_get_client, patch(
        "backend.executor.manager.settings"
    ) as mock_settings:

        # Setup mocks
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_settings.config.low_balance_threshold = 500  # $5 threshold

        # Create mock database client
        mock_db_client = MagicMock()

        # Test the low balance handler
        execution_processor._handle_low_balance(
            db_client=mock_db_client,
            user_id=user_id,
            current_balance=current_balance,
            transaction_cost=transaction_cost,
        )

        # Verify no notification was sent
        mock_queue_notif.assert_not_called()
        mock_client.discord_system_alert.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_handle_low_balance_no_duplicate_when_already_below(
    server: SpinTestServer,
):
    """Test that no notification is sent when already below threshold."""

    execution_processor = ExecutionProcessor()
    user_id = "test-user-123"
    current_balance = 300  # $3 - below $5 threshold
    transaction_cost = (
        100  # $1 transaction (balance before was $4, also below threshold)
    )

    # Mock dependencies
    with patch(
        "backend.executor.manager.queue_notification"
    ) as mock_queue_notif, patch(
        "backend.executor.manager.get_notification_manager_client"
    ) as mock_get_client, patch(
        "backend.executor.manager.settings"
    ) as mock_settings:

        # Setup mocks
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_settings.config.low_balance_threshold = 500  # $5 threshold

        # Create mock database client
        mock_db_client = MagicMock()

        # Test the low balance handler
        execution_processor._handle_low_balance(
            db_client=mock_db_client,
            user_id=user_id,
            current_balance=current_balance,
            transaction_cost=transaction_cost,
        )

        # Verify no notification was sent (user was already below threshold)
        mock_queue_notif.assert_not_called()
        mock_client.discord_system_alert.assert_not_called()
