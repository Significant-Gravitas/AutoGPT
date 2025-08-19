from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import NotificationType
from prisma.models import CreditTransaction

from backend.data.credit import BetaUserCredit, UsageTransactionMetadata
from backend.data.notifications import LowBalanceData
from backend.data.user import DEFAULT_USER_ID
from backend.util.test import SpinTestServer


async def cleanup_test_user_transactions():
    """Clean up any existing transactions for the test user."""
    await CreditTransaction.prisma().delete_many(where={"userId": DEFAULT_USER_ID})


@pytest.mark.asyncio(loop_scope="session")
async def test_low_balance_threshold_notification(server: SpinTestServer):
    """Test that LOW_BALANCE notification is triggered when crossing threshold."""

    # Setup
    await cleanup_test_user_transactions()
    user_credit = BetaUserCredit(2000)  # $20 refill value
    user_id = DEFAULT_USER_ID

    # Mock the notification queue, Discord client, and user email lookup
    from unittest.mock import MagicMock

    with patch(
        "backend.data.credit.queue_notification_async", new_callable=AsyncMock
    ) as mock_queue_notif, patch(
        "backend.data.credit.get_notification_manager_client"
    ) as mock_get_client, patch(
        "backend.data.credit.get_user_email_by_id", new_callable=AsyncMock
    ) as mock_get_email:

        # Create a mock client with discord_system_alert method
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_get_email.return_value = "test@example.com"

        # Set return value for the async notification queue
        mock_queue_notif.return_value = {"success": True}

        # Don't add extra balance - BetaUserCredit will auto-top up to $20
        # on first get_credits() call
        balance_initial = await user_credit.get_credits(user_id)
        print(f"Initial balance: ${balance_initial/100}")

        # Spend to cross the $5 threshold (default is 500 = $5)
        # This should trigger a LOW_BALANCE notification
        await user_credit.spend_credits(
            user_id,
            1600,  # Spend $16, leaving $4 (below $5 threshold)
            metadata=UsageTransactionMetadata(reason="test"),
        )

        # Verify notification was queued
        mock_queue_notif.assert_called_once()
        notification_call = mock_queue_notif.call_args[0][0]

        # Check it's a LOW_BALANCE notification
        assert notification_call.type == NotificationType.LOW_BALANCE
        assert notification_call.user_id == user_id
        assert isinstance(notification_call.data, LowBalanceData)
        assert notification_call.data.current_balance == 400  # $4 in credits

        # Verify Discord alert was sent
        mock_client.discord_system_alert.assert_called_once()
        discord_message = mock_client.discord_system_alert.call_args[0][0]
        assert "Low Balance Alert" in discord_message
        # User ID or email should be in message
        assert user_id in discord_message or "test@example.com" in discord_message
        assert "$4.00" in discord_message


@pytest.mark.asyncio(loop_scope="session")
async def test_low_balance_no_duplicate_notification(server: SpinTestServer):
    """Test that LOW_BALANCE notification is not sent repeatedly."""

    # Setup
    await cleanup_test_user_transactions()
    user_credit = BetaUserCredit(1500)  # $15 refill value
    user_id = DEFAULT_USER_ID

    # Mock the notification queue
    from unittest.mock import MagicMock

    with patch(
        "backend.data.credit.queue_notification_async", new_callable=AsyncMock
    ) as mock_queue_notif, patch(
        "backend.data.credit.get_notification_manager_client"
    ) as mock_get_client, patch(
        "backend.data.credit.get_user_email_by_id", new_callable=AsyncMock
    ) as mock_get_email:

        # Create a mock client with discord_system_alert method
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_get_email.return_value = "test@example.com"

        # Set return value for the async notification queue
        mock_queue_notif.return_value = {"success": True}

        # Don't add extra balance - BetaUserCredit will auto-top up to $15
        # on first get_credits() call

        # First spend to cross threshold
        balance_before = await user_credit.get_credits(user_id)
        print(f"Balance before spend: ${balance_before/100}")

        await user_credit.spend_credits(
            user_id,
            1100,  # Spend $11, leaving $4 (below $5 threshold)
            metadata=UsageTransactionMetadata(reason="test"),
        )

        balance_after = await user_credit.get_credits(user_id)
        print(f"Balance after spend: ${balance_after/100}")
        print(f"Mock called {mock_queue_notif.call_count} times")

        # Should have sent notification
        assert mock_queue_notif.call_count == 1

        # Spend again below threshold
        await user_credit.spend_credits(
            user_id,
            100,  # Spend $1, leaving $8
            metadata=UsageTransactionMetadata(reason="test"),
        )

        # Should NOT send another notification
        assert mock_queue_notif.call_count == 1
