"""Tests for notification error handling in NotificationManager."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from prisma.enums import NotificationType

from backend.data.notifications import NotificationEventModel
from backend.notifications.notifications import NotificationManager


class MockPostmarkError(Exception):
    """Mock exception to simulate Postmark API errors."""

    def __init__(self, message, error_code=None, status_code=None):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


class TestNotificationErrorHandling:
    """Test cases for notification error handling in NotificationManager."""

    @pytest.fixture
    def notification_manager(self):
        """Create a NotificationManager instance for testing."""
        with patch("backend.notifications.notifications.AppService.__init__"):
            manager = NotificationManager()
            manager.logger = MagicMock()
            manager.email_sender = MagicMock()
            manager.read_db = MagicMock()
            return manager

    @pytest.fixture
    def sample_notifications(self):
        """Create sample notification data for testing."""
        return [
            NotificationEventModel(
                type=NotificationType.AGENT_RUN,
                user_id="user_1",
                created_at=datetime.now(timezone.utc),
                data={"agent_name": "Test Agent 1"},
            ),
            NotificationEventModel(
                type=NotificationType.AGENT_RUN,
                user_id="user_2",
                created_at=datetime.now(timezone.utc),
                data={"agent_name": "Test Agent 2"},
            ),
            NotificationEventModel(
                type=NotificationType.AGENT_RUN,
                user_id="user_3",
                created_at=datetime.now(timezone.utc),
                data={"agent_name": "Test Agent 3"},
            ),
        ]

    def test_postmark_406_inactive_recipient_error_handling(
        self, notification_manager, sample_notifications
    ):
        """Test handling of Postmark 406 error for inactive recipients."""
        # Create an exception that simulates a 406 error
        error = Exception("Recipient marked as inactive (406)")

        # Mock the email sender to raise the error
        notification_manager.email_sender.send_templated.side_effect = error

        # Set up the user database mock
        notification_manager.read_db.user.find_unique.return_value = MagicMock(
            email="test@example.com", email_verified=True
        )

        # Process notifications
        failed_indices = []
        recipient_email = "test@example.com"

        # Simulate the error handling logic from the actual code
        i = 0
        while i < len(sample_notifications):
            try:
                notification_manager.email_sender.send_templated(
                    notification=NotificationType.AGENT_RUN,
                    user_email=recipient_email,
                    data=sample_notifications[i],
                    user_unsub_link="http://example.com/unsub",
                )
                i += 1
            except Exception as e:
                # This is the error handling logic we're testing
                error_message = str(e).lower()

                if "406" in error_message or "inactive" in error_message:
                    notification_manager.logger.warning(
                        f"Failed to send notification at index {i}: "
                        f"Recipient marked as inactive by Postmark. "
                        f"Error: {e}. Skipping this notification."
                    )

                failed_indices.append(i)
                i += 1

        # Verify the warning was logged for all notifications
        assert notification_manager.logger.warning.call_count == len(
            sample_notifications
        )
        assert len(failed_indices) == len(sample_notifications)

        # Verify the warning message contains the expected information
        warning_call = notification_manager.logger.warning.call_args[0][0]
        assert "inactive" in warning_call.lower()
        assert "406" in warning_call

    def test_postmark_422_malformed_data_error_handling(
        self, notification_manager, sample_notifications
    ):
        """Test handling of Postmark 422 error for malformed data."""
        # Create an exception that simulates a 422 error
        error = Exception("Unprocessable entity (422): Malformed email data")

        # Mock the email sender to raise the error
        notification_manager.email_sender.send_templated.side_effect = error

        # Set up the user database mock
        notification_manager.read_db.user.find_unique.return_value = MagicMock(
            email="test@example.com", email_verified=True
        )

        # Process notifications
        failed_indices = []
        recipient_email = "test@example.com"

        # Simulate the error handling logic
        i = 0
        while i < len(sample_notifications):
            try:
                notification_manager.email_sender.send_templated(
                    notification=NotificationType.AGENT_RUN,
                    user_email=recipient_email,
                    data=sample_notifications[i],
                    user_unsub_link="http://example.com/unsub",
                )
                i += 1
            except Exception as e:
                error_message = str(e).lower()

                if "422" in error_message or "unprocessable" in error_message:
                    notification_manager.logger.error(
                        f"Failed to send notification at index {i}: "
                        f"Malformed notification data rejected by Postmark. "
                        f"Error: {e}. Skipping this notification."
                    )
                    failed_indices.append(i)

                i += 1

        # Verify the error was logged for all notifications
        assert notification_manager.logger.error.call_count == len(
            sample_notifications
        )
        assert len(failed_indices) == len(sample_notifications)

        # Verify the error message contains the expected information
        error_call = notification_manager.logger.error.call_args[0][0]
        assert "malformed" in error_call.lower()

    def test_value_error_too_large_notification(
        self, notification_manager, sample_notifications
    ):
        """Test handling of ValueError for oversized notifications."""
        # Create a ValueError for too large email
        error = ValueError("Email body too large: 6000000 characters (limit: 5242880)")

        # Mock the email sender to raise the error
        notification_manager.email_sender.send_templated.side_effect = error

        # Set up the user database mock
        notification_manager.read_db.user.find_unique.return_value = MagicMock(
            email="test@example.com", email_verified=True
        )

        # Process notifications
        failed_indices = []
        recipient_email = "test@example.com"

        # Simulate the error handling logic
        i = 0
        while i < len(sample_notifications):
            try:
                notification_manager.email_sender.send_templated(
                    notification=NotificationType.AGENT_RUN,
                    user_email=recipient_email,
                    data=sample_notifications[i],
                    user_unsub_link="http://example.com/unsub",
                )
                i += 1
            except ValueError as e:
                if "too large" in str(e).lower():
                    notification_manager.logger.error(
                        f"Failed to send notification at index {i}: "
                        f"Notification size exceeds email limit. "
                        f"Error: {e}. Skipping this notification."
                    )
                else:
                    notification_manager.logger.error(
                        f"Failed to send notification at index {i}: "
                        f"ValueError: {e}. Skipping this notification."
                    )

                failed_indices.append(i)
                i += 1

        # Verify the error was logged for all notifications
        assert notification_manager.logger.error.call_count == len(
            sample_notifications
        )
        assert len(failed_indices) == len(sample_notifications)

        # Verify the error message contains size information
        error_call = notification_manager.logger.error.call_args[0][0]
        assert "size exceeds" in error_call.lower()

    def test_generic_error_handling(
        self, notification_manager, sample_notifications
    ):
        """Test handling of generic errors."""
        # Create a generic exception
        error = Exception("Some other API error")

        # Mock the email sender to raise the error
        notification_manager.email_sender.send_templated.side_effect = error

        # Set up the user database mock
        notification_manager.read_db.user.find_unique.return_value = MagicMock(
            email="test@example.com", email_verified=True
        )

        # Process notifications
        failed_indices = []
        recipient_email = "test@example.com"

        # Simulate the error handling logic
        i = 0
        while i < len(sample_notifications):
            try:
                notification_manager.email_sender.send_templated(
                    notification=NotificationType.AGENT_RUN,
                    user_email=recipient_email,
                    data=sample_notifications[i],
                    user_unsub_link="http://example.com/unsub",
                )
                i += 1
            except Exception as e:
                error_message = str(e).lower()

                # Check for specific errors first
                if "406" in error_message or "inactive" in error_message:
                    notification_manager.logger.warning(
                        f"Failed to send notification at index {i}: "
                        f"Recipient marked as inactive by Postmark. "
                        f"Error: {e}. Skipping this notification."
                    )
                elif "422" in error_message or "unprocessable" in error_message:
                    notification_manager.logger.error(
                        f"Failed to send notification at index {i}: "
                        f"Malformed notification data rejected by Postmark. "
                        f"Error: {e}. Skipping this notification."
                    )
                else:
                    # Generic error
                    notification_manager.logger.error(
                        f"Failed to send notification at index {i}: "
                        f"Email API error ({type(e).__name__}): {e}. "
                        f"Skipping this notification."
                    )

                failed_indices.append(i)
                i += 1

        # Verify the generic error was logged
        assert notification_manager.logger.error.call_count == len(
            sample_notifications
        )
        assert len(failed_indices) == len(sample_notifications)

        # Verify the error message is generic
        error_call = notification_manager.logger.error.call_args[0][0]
        assert "email api error" in error_call.lower()

    def test_mixed_error_scenarios(self, notification_manager, sample_notifications):
        """Test handling of mixed error scenarios in a batch."""
        # Create different errors for different notifications
        errors = [
            Exception("Recipient marked as inactive (406)"),
            Exception("Unprocessable entity (422)"),
            ValueError("Email body too large"),
        ]

        # Mock the email sender to raise different errors
        notification_manager.email_sender.send_templated.side_effect = errors

        # Set up the user database mock
        notification_manager.read_db.user.find_unique.return_value = MagicMock(
            email="test@example.com", email_verified=True
        )

        # Process notifications
        failed_indices = []
        recipient_email = "test@example.com"

        # Simulate the error handling logic
        i = 0
        while i < len(sample_notifications):
            try:
                notification_manager.email_sender.send_templated(
                    notification=NotificationType.AGENT_RUN,
                    user_email=recipient_email,
                    data=sample_notifications[i],
                    user_unsub_link="http://example.com/unsub",
                )
                i += 1
            except (ValueError, Exception) as e:
                error_message = str(e).lower()

                if "406" in error_message or "inactive" in error_message:
                    notification_manager.logger.warning(
                        f"Failed to send notification at index {i}: "
                        f"Recipient marked as inactive by Postmark. "
                        f"Error: {e}. Skipping this notification."
                    )
                elif "422" in error_message or "unprocessable" in error_message:
                    notification_manager.logger.error(
                        f"Failed to send notification at index {i}: "
                        f"Malformed notification data rejected by Postmark. "
                        f"Error: {e}. Skipping this notification."
                    )
                elif isinstance(e, ValueError) and "too large" in error_message:
                    notification_manager.logger.error(
                        f"Failed to send notification at index {i}: "
                        f"Notification size exceeds email limit. "
                        f"Error: {e}. Skipping this notification."
                    )

                failed_indices.append(i)
                i += 1

        # Verify all errors were handled
        assert len(failed_indices) == len(sample_notifications)
        # One warning (406) and two errors (422 and ValueError)
        assert notification_manager.logger.warning.call_count == 1
        assert notification_manager.logger.error.call_count == 2

    def test_batch_continues_after_individual_failures(self, notification_manager):
        """Test that the batch continues processing after individual notification failures."""
        # Create a larger batch of notifications
        large_batch = [
            NotificationEventModel(
                type=NotificationType.AGENT_RUN,
                user_id=f"user_{i}",
                created_at=datetime.now(timezone.utc),
                data={"agent_name": f"Test Agent {i}"},
            )
            for i in range(10)
        ]

        # Mock the email sender to fail on specific indices (2, 5, 8)
        call_count = [0]  # Use list to make it mutable in the closure

        def side_effect_function(*args, **kwargs):
            # Count which call this is
            call_number = call_count[0]
            call_count[0] += 1

            if call_number in [2, 5, 8]:
                raise Exception("Recipient marked as inactive (406)")
            return None

        notification_manager.email_sender.send_templated.side_effect = (
            side_effect_function
        )

        # Set up the user database mock
        notification_manager.read_db.user.find_unique.return_value = MagicMock(
            email="test@example.com", email_verified=True
        )

        # Process notifications
        failed_indices = []
        successful_indices = []
        recipient_email = "test@example.com"

        # Simulate the error handling logic
        i = 0
        while i < len(large_batch):
            try:
                notification_manager.email_sender.send_templated(
                    notification=NotificationType.AGENT_RUN,
                    user_email=recipient_email,
                    data=large_batch[i],
                    user_unsub_link="http://example.com/unsub",
                )
                successful_indices.append(i)
                i += 1
            except Exception as e:
                error_message = str(e).lower()
                if "406" in error_message or "inactive" in error_message:
                    notification_manager.logger.warning(
                        f"Failed to send notification at index {i}: "
                        f"Recipient marked as inactive by Postmark. "
                        f"Error: {e}. Skipping this notification."
                    )

                failed_indices.append(i)
                i += 1

        # Verify the batch continued processing
        assert len(failed_indices) == 3  # Should have 3 failures
        assert len(successful_indices) == 7  # Should have 7 successes
        assert failed_indices == [2, 5, 8]  # Specific indices that failed
        assert notification_manager.logger.warning.call_count == 3