"""Tests for notification error handling in NotificationManager."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from prisma.enums import NotificationType

from backend.data.notifications import AgentRunData, NotificationEventModel
from backend.notifications.notifications import NotificationManager


class TestNotificationErrorHandling:
    """Test cases for notification error handling in NotificationManager."""

    @pytest.fixture
    def notification_manager(self):
        """Create a NotificationManager instance for testing."""
        with patch("backend.notifications.notifications.AppService.__init__"):
            manager = NotificationManager()
            manager.email_sender = MagicMock()
            # Mock the _get_template method used by _process_batch
            template_mock = Mock()
            template_mock.base_template = "base"
            template_mock.subject_template = "subject"
            template_mock.body_template = "body"
            manager.email_sender._get_template = Mock(return_value=template_mock)
            # Mock the formatter
            manager.email_sender.formatter = Mock()
            manager.email_sender.formatter.format_email = Mock(
                return_value=("subject", "body content")
            )
            manager.email_sender.formatter.env = Mock()
            manager.email_sender.formatter.env.globals = {
                "base_url": "http://example.com"
            }
            return manager

    @pytest.fixture
    def sample_batch_event(self):
        """Create a sample batch event for testing."""
        return NotificationEventModel(
            type=NotificationType.AGENT_RUN,
            user_id="user_1",
            created_at=datetime.now(timezone.utc),
            data=AgentRunData(
                agent_name="Test Agent",
                credits_used=10.0,
                execution_time=5.0,
                node_count=3,
                graph_id="graph_1",
                outputs=[],
            ),
        )

    @pytest.fixture
    def sample_batch_notifications(self):
        """Create sample batch notifications for testing."""
        notifications = []
        for i in range(3):
            notification = Mock()
            notification.type = NotificationType.AGENT_RUN
            notification.data = {
                "agent_name": f"Test Agent {i}",
                "credits_used": 10.0 * (i + 1),
                "execution_time": 5.0 * (i + 1),
                "node_count": 3 + i,
                "graph_id": f"graph_{i}",
                "outputs": [],
            }
            notification.created_at = datetime.now(timezone.utc)
            notifications.append(notification)
        return notifications

    @pytest.mark.asyncio
    async def test_postmark_406_inactive_recipient_error_handling(
        self, notification_manager, sample_batch_event, sample_batch_notifications
    ):
        """Test handling of Postmark 406 error for inactive recipients and email deactivation."""
        with patch("backend.notifications.notifications.logger") as mock_logger, patch(
            "backend.notifications.notifications.set_user_email_verification",
            new_callable=AsyncMock,
        ) as mock_set_verification, patch(
            "backend.notifications.notifications.get_database_manager_async_client"
        ) as mock_db_client, patch(
            "backend.notifications.notifications.generate_unsubscribe_link"
        ) as mock_unsub_link:

            # Setup mocks
            mock_db = mock_db_client.return_value
            mock_db.get_user_email_by_id = AsyncMock(return_value="test@example.com")
            mock_db.get_user_notification_batch = AsyncMock(
                return_value=Mock(notifications=sample_batch_notifications)
            )
            mock_db.empty_user_notification_batch = AsyncMock()
            mock_unsub_link.return_value = "http://example.com/unsub"

            # Mock _should_email_user_based_on_preference
            notification_manager._should_email_user_based_on_preference = AsyncMock(
                return_value=True
            )
            notification_manager._should_batch = AsyncMock(return_value=True)
            notification_manager._parse_message = Mock(return_value=sample_batch_event)

            # Configure email sender to raise 406 error for single notifications
            def send_side_effect(*args, **kwargs):
                # Check if data is a list (batch) or single notification
                data = kwargs.get("data", [])
                if isinstance(data, list) and len(data) == 1:
                    raise Exception("Recipient marked as inactive (406)")
                raise Exception("Recipient marked as inactive (406)")

            notification_manager.email_sender.send_templated.side_effect = (
                send_side_effect
            )

            # Create message
            message = sample_batch_event.model_dump_json()

            # Act: call actual implementation
            result = await notification_manager._process_batch(message)

            # Assert: processing returns success (True = handled, won't retry)
            assert result is True

            # Email verification should be set to false due to 406
            assert mock_set_verification.called
            mock_set_verification.assert_any_call("user_1", False)

            # Warning logs should be emitted for inactive recipient
            warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
            assert any("inactive" in call.lower() for call in warning_calls)
            assert any("406" in call for call in warning_calls)

    @pytest.mark.asyncio
    async def test_postmark_422_malformed_data_error_handling(
        self, notification_manager, sample_batch_event, sample_batch_notifications
    ):
        """Test handling of Postmark 422 error for malformed data."""
        with patch("backend.notifications.notifications.logger") as mock_logger, patch(
            "backend.notifications.notifications.get_database_manager_async_client"
        ) as mock_db_client, patch(
            "backend.notifications.notifications.generate_unsubscribe_link"
        ) as mock_unsub_link:

            # Setup mocks
            mock_db = mock_db_client.return_value
            mock_db.get_user_email_by_id = AsyncMock(return_value="test@example.com")
            mock_db.get_user_notification_batch = AsyncMock(
                return_value=Mock(notifications=sample_batch_notifications)
            )
            mock_db.empty_user_notification_batch = AsyncMock()
            mock_unsub_link.return_value = "http://example.com/unsub"

            # Mock internal methods
            notification_manager._should_email_user_based_on_preference = AsyncMock(
                return_value=True
            )
            notification_manager._should_batch = AsyncMock(return_value=True)
            notification_manager._parse_message = Mock(return_value=sample_batch_event)

            # Configure email sender to raise 422 error
            def send_side_effect(*args, **kwargs):
                data = kwargs.get("data", [])
                if isinstance(data, list) and len(data) == 1:
                    raise Exception("Unprocessable entity (422): Malformed email data")
                raise Exception("Unprocessable entity (422): Malformed email data")

            notification_manager.email_sender.send_templated.side_effect = (
                send_side_effect
            )

            # Create message
            message = sample_batch_event.model_dump_json()

            # Act
            result = await notification_manager._process_batch(message)

            # Assert
            assert result is True

            # Error logs should be emitted for malformed data
            error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
            assert any("malformed" in call.lower() for call in error_calls)
            assert any("422" in call for call in error_calls)

    @pytest.mark.asyncio
    async def test_postmark_oversized_notification_error_handling(
        self, notification_manager, sample_batch_event, sample_batch_notifications
    ):
        """Test handling of ValueError for oversized notifications."""
        with patch("backend.notifications.notifications.logger") as mock_logger, patch(
            "backend.notifications.notifications.get_database_manager_async_client"
        ) as mock_db_client, patch(
            "backend.notifications.notifications.generate_unsubscribe_link"
        ) as mock_unsub_link:

            # Setup mocks
            mock_db = mock_db_client.return_value
            mock_db.get_user_email_by_id = AsyncMock(return_value="test@example.com")
            mock_db.get_user_notification_batch = AsyncMock(
                return_value=Mock(notifications=sample_batch_notifications)
            )
            mock_db.empty_user_notification_batch = AsyncMock()
            mock_unsub_link.return_value = "http://example.com/unsub"

            # Mock internal methods
            notification_manager._should_email_user_based_on_preference = AsyncMock(
                return_value=True
            )
            notification_manager._should_batch = AsyncMock(return_value=True)
            notification_manager._parse_message = Mock(return_value=sample_batch_event)

            # Configure email sender to raise ValueError for oversized email
            def send_side_effect(*args, **kwargs):
                data = kwargs.get("data", [])
                if isinstance(data, list) and len(data) == 1:
                    raise ValueError(
                        "Email body too large: 6000000 characters (limit: 5242880)"
                    )
                raise ValueError(
                    "Email body too large: 6000000 characters (limit: 5242880)"
                )

            notification_manager.email_sender.send_templated.side_effect = (
                send_side_effect
            )

            # Create message
            message = sample_batch_event.model_dump_json()

            # Act
            result = await notification_manager._process_batch(message)

            # Assert
            assert result is True

            # Error logs should be emitted for oversized notification
            error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
            assert any(
                "size exceeds" in call.lower() or "too large" in call.lower()
                for call in error_calls
            )

    @pytest.mark.asyncio
    async def test_postmark_generic_api_error_handling(
        self, notification_manager, sample_batch_event, sample_batch_notifications
    ):
        """Test handling of generic API errors."""
        with patch("backend.notifications.notifications.logger") as mock_logger, patch(
            "backend.notifications.notifications.get_database_manager_async_client"
        ) as mock_db_client, patch(
            "backend.notifications.notifications.generate_unsubscribe_link"
        ) as mock_unsub_link:

            # Setup mocks
            mock_db = mock_db_client.return_value
            mock_db.get_user_email_by_id = AsyncMock(return_value="test@example.com")
            mock_db.get_user_notification_batch = AsyncMock(
                return_value=Mock(notifications=sample_batch_notifications)
            )
            mock_db.empty_user_notification_batch = AsyncMock()
            mock_unsub_link.return_value = "http://example.com/unsub"

            # Mock internal methods
            notification_manager._should_email_user_based_on_preference = AsyncMock(
                return_value=True
            )
            notification_manager._should_batch = AsyncMock(return_value=True)
            notification_manager._parse_message = Mock(return_value=sample_batch_event)

            # Configure email sender to raise generic error
            def send_side_effect(*args, **kwargs):
                data = kwargs.get("data", [])
                if isinstance(data, list) and len(data) == 1:
                    raise Exception("Some other API error")
                raise Exception("Some other API error")

            notification_manager.email_sender.send_templated.side_effect = (
                send_side_effect
            )

            # Create message
            message = sample_batch_event.model_dump_json()

            # Act
            result = await notification_manager._process_batch(message)

            # Assert
            assert result is True

            # Generic error logs should be emitted
            error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
            assert any(
                "email api error" in call.lower() or "skipping" in call.lower()
                for call in error_calls
            )

    @pytest.mark.asyncio
    async def test_postmark_batch_processing_with_mixed_errors(
        self, notification_manager, sample_batch_event
    ):
        """Test handling of mixed error scenarios in a batch."""
        with patch("backend.notifications.notifications.logger") as mock_logger, patch(
            "backend.notifications.notifications.set_user_email_verification",
            new_callable=AsyncMock,
        ) as mock_set_verification, patch(
            "backend.notifications.notifications.get_database_manager_async_client"
        ) as mock_db_client, patch(
            "backend.notifications.notifications.generate_unsubscribe_link"
        ) as mock_unsub_link:

            # Create larger batch with mixed notifications
            notifications = []
            for i in range(5):
                notification = Mock()
                notification.type = NotificationType.AGENT_RUN
                notification.data = {
                    "agent_name": f"Test Agent {i}",
                    "credits_used": 10.0 * (i + 1),
                    "execution_time": 5.0 * (i + 1),
                    "node_count": 3 + i,
                    "graph_id": f"graph_{i}",
                    "outputs": [],
                }
                notification.created_at = datetime.now(timezone.utc)
                notifications.append(notification)

            # Setup mocks
            mock_db = mock_db_client.return_value
            mock_db.get_user_email_by_id = AsyncMock(return_value="test@example.com")
            mock_db.get_user_notification_batch = AsyncMock(
                return_value=Mock(notifications=notifications)
            )
            mock_db.empty_user_notification_batch = AsyncMock()
            mock_unsub_link.return_value = "http://example.com/unsub"

            # Mock internal methods
            notification_manager._should_email_user_based_on_preference = AsyncMock(
                return_value=True
            )
            notification_manager._should_batch = AsyncMock(return_value=True)
            notification_manager._parse_message = Mock(return_value=sample_batch_event)

            # Track call count to return different errors
            call_count = [0]

            def send_side_effect(*args, **kwargs):
                data = kwargs.get("data", [])
                # Only fail for single notifications
                if isinstance(data, list) and len(data) == 1:
                    current_call = call_count[0]
                    call_count[0] += 1

                    if current_call == 0:
                        raise Exception("Recipient marked as inactive (406)")
                    elif current_call == 1:
                        raise Exception("Unprocessable entity (422)")
                    elif current_call == 2:
                        raise ValueError("Email body too large")
                    elif current_call == 3:
                        raise Exception("Generic API error")
                    else:
                        # Let the last one succeed
                        return None
                # For batch attempts, fail to force single processing
                raise Exception("Force single processing")

            notification_manager.email_sender.send_templated.side_effect = (
                send_side_effect
            )

            # Create message
            message = sample_batch_event.model_dump_json()

            # Act
            result = await notification_manager._process_batch(message)

            # Assert
            assert result is True

            # Verify mixed error handling
            assert mock_logger.warning.called  # For 406
            assert mock_logger.error.called  # For 422, ValueError, generic
            assert mock_set_verification.called  # For 406 error

    @pytest.mark.asyncio
    async def test_batch_continues_after_individual_failures(
        self, notification_manager, sample_batch_event
    ):
        """Test that the batch continues processing after individual notification failures."""
        with patch("backend.notifications.notifications.logger") as mock_logger, patch(
            "backend.notifications.notifications.get_database_manager_async_client"
        ) as mock_db_client, patch(
            "backend.notifications.notifications.generate_unsubscribe_link"
        ) as mock_unsub_link:

            # Create larger batch
            notifications = []
            for i in range(10):
                notification = Mock()
                notification.type = NotificationType.AGENT_RUN
                notification.data = {
                    "agent_name": f"Test Agent {i}",
                    "credits_used": 10.0 * (i + 1),
                    "execution_time": 5.0 * (i + 1),
                    "node_count": 3 + i,
                    "graph_id": f"graph_{i}",
                    "outputs": [],
                }
                notification.created_at = datetime.now(timezone.utc)
                notifications.append(notification)

            # Setup mocks
            mock_db = mock_db_client.return_value
            mock_db.get_user_email_by_id = AsyncMock(return_value="test@example.com")
            mock_db.get_user_notification_batch = AsyncMock(
                return_value=Mock(notifications=notifications)
            )
            mock_db.empty_user_notification_batch = AsyncMock()
            mock_unsub_link.return_value = "http://example.com/unsub"

            # Mock internal methods
            notification_manager._should_email_user_based_on_preference = AsyncMock(
                return_value=True
            )
            notification_manager._should_batch = AsyncMock(return_value=True)
            notification_manager._parse_message = Mock(return_value=sample_batch_event)

            # Track calls
            call_count = [0]
            successful_calls = []
            failed_calls = []

            def send_side_effect(*args, **kwargs):
                data = kwargs.get("data", [])
                # Only process single notifications
                if isinstance(data, list) and len(data) == 1:
                    current_call = call_count[0]
                    call_count[0] += 1

                    # Fail on indices 2, 5, 8
                    if current_call in [2, 5, 8]:
                        failed_calls.append(current_call)
                        raise Exception("Recipient marked as inactive (406)")
                    else:
                        successful_calls.append(current_call)
                        return None
                # Force single processing
                raise Exception("Force single processing")

            notification_manager.email_sender.send_templated.side_effect = (
                send_side_effect
            )

            # Create message
            message = sample_batch_event.model_dump_json()

            # Act
            result = await notification_manager._process_batch(message)

            # Assert
            assert result is True

            # Verify batch processing continued despite failures
            assert len(failed_calls) == 3  # Should have 3 failures
            assert len(successful_calls) == 7  # Should have 7 successes
            assert failed_calls == [2, 5, 8]  # Specific indices that failed

            # Verify warnings were logged for failures
            assert mock_logger.warning.call_count >= 3

            # Batch should not be emptied since not all were successful
            assert not mock_db.empty_user_notification_batch.called
