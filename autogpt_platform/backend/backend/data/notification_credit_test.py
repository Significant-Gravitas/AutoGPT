"""Test notification behavior for credit balance alerts."""

from unittest.mock import MagicMock, patch

import pytest
from prisma.enums import CreditTransactionType, NotificationType

from backend.data.credit import BetaUserCredit, UsageTransactionMetadata
from backend.data.notifications import LowBalanceData
from backend.data.user import DEFAULT_USER_ID
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_low_balance_threshold_notification(server: SpinTestServer):
    """Test that LOW_BALANCE notification is triggered when crossing threshold."""

    # Setup
    user_credit = BetaUserCredit(2000)  # $20 starting balance
    user_id = DEFAULT_USER_ID

    # Mock the notification queue and Discord client
    with patch(
        "backend.notifications.notifications.queue_notification_async"
    ) as mock_queue_notif, patch(
        "backend.util.clients.get_notification_manager_client"
    ) as mock_discord:

        mock_discord_client = MagicMock()
        mock_discord.return_value = mock_discord_client

        # Start with balance above threshold ($20)
        await user_credit._add_transaction(
            user_id,
            2000,  # $20
            CreditTransactionType.TOP_UP,
        )

        # Spend to cross the $10 threshold
        # This should trigger a LOW_BALANCE notification
        await user_credit.spend_credits(
            user_id,
            1100,  # Spend $11, leaving $9
            metadata=UsageTransactionMetadata(reason="test"),
        )

        # Verify notification was queued
        mock_queue_notif.assert_called_once()
        notification_call = mock_queue_notif.call_args[0][0]

        # Check it's a LOW_BALANCE notification
        assert notification_call.type == NotificationType.LOW_BALANCE
        assert notification_call.user_id == user_id
        assert isinstance(notification_call.data, LowBalanceData)
        assert notification_call.data.current_balance == 900  # $9 in credits

        # Verify Discord alert was sent
        mock_discord_client.discord_system_alert.assert_called_once()
        discord_message = mock_discord_client.discord_system_alert.call_args[0][0]
        assert "LOW BALANCE" in discord_message
        assert user_id in discord_message
        assert "$9.00" in discord_message


@pytest.mark.asyncio
async def test_low_balance_no_duplicate_notification():
    """Test that LOW_BALANCE notification is not sent repeatedly."""

    # Setup
    user_credit = BetaUserCredit(1500)  # $15 starting balance
    user_id = DEFAULT_USER_ID

    # Mock the notification queue
    with patch("backend.data.credit.queue_notification") as mock_queue_notif, patch(
        "backend.data.credit.get_notification_manager_client"
    ) as mock_discord:

        mock_discord_client = MagicMock()
        mock_discord.return_value = mock_discord_client

        # Start with balance above threshold
        await user_credit._add_transaction(
            user_id,
            1500,  # $15
            CreditTransactionType.TOP_UP,
        )

        # First spend to cross threshold
        await user_credit.spend_credits(
            user_id,
            600,  # Spend $6, leaving $9 (below $10 threshold)
            metadata=UsageTransactionMetadata(reason="test"),
        )

        # Should trigger notification
        assert mock_queue_notif.call_count == 1
        assert mock_discord_client.discord_system_alert.call_count == 1

        # Reset mocks
        mock_queue_notif.reset_mock()
        mock_discord_client.discord_system_alert.reset_mock()

        # Second spend while still below threshold
        await user_credit.spend_credits(
            user_id,
            100,  # Spend $1, leaving $8
            metadata=UsageTransactionMetadata(reason="test"),
        )

        # Should NOT trigger another notification (already below threshold)
        mock_queue_notif.assert_not_called()
        mock_discord_client.discord_system_alert.assert_not_called()


# @pytest.mark.asyncio
# async def test_insufficient_funds_notification():
#     """Test that ZERO_BALANCE notification is triggered on insufficient funds."""
#     from backend.data.credit import InsufficientBalanceError
#     from backend.executor.manager import ExecutionManager

#     # Setup
#     user_credit = BetaUserCredit(100)  # $1 starting balance
#     user_id = DEFAULT_USER_ID

#     # Mock dependencies
#     with patch(
#         "backend.executor.manager.queue_notification"
#     ) as mock_queue_notif, patch(
#         "backend.executor.manager.get_notification_manager_client"
#     ) as mock_discord, patch(
#         "backend.executor.manager.ExecInput"
#     ) as mock_exec_input:

#         mock_discord_client = MagicMock()
#         mock_discord.return_value = mock_discord_client

#         # Create manager instance
#         manager = ExecutionManager()

#         # Mock db_client
#         mock_db_client = AsyncMock()
#         mock_db_client.get_user_by_id = AsyncMock(
#             return_value=MagicMock(id=user_id, email="test@example.com")
#         )

#         # Setup insufficient funds error
#         error = InsufficientBalanceError(
#             user_id=user_id,
#             message="Insufficient funds for operation",
#             balance=50,
#             amount=200,
#         )

#         # Call the handler
#         await manager._handle_insufficient_funds_notif(
#             db_client=mock_db_client,
#             user_id=user_id,
#             graph_id="test_graph",
#             exec_stats={"duration": 1.0},
#             e=error,
#         )

#         # Verify ZERO_BALANCE notification was queued
#         mock_queue_notif.assert_called_once()
#         notification_call = mock_queue_notif.call_args[0][0]

#         # Check it's a ZERO_BALANCE notification
#         assert notification_call.type == NotificationType.ZERO_BALANCE
#         assert notification_call.user_id == user_id
#         assert isinstance(notification_call.data, ZeroBalanceData)

#         # Verify Discord alert was sent
#         mock_discord_client.discord_system_alert.assert_called_once()
#         discord_message = mock_discord_client.discord_system_alert.call_args[0][0]
#         assert "INSUFFICIENT FUNDS" in discord_message
#         assert "test@example.com" in discord_message
