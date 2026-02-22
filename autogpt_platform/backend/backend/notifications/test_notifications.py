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
    async def test_406_stops_all_processing_for_user(
        self, notification_manager, sample_batch_event
    ):
        """Test that 406 inactive recipient error stops ALL processing for that user."""
        with patch("backend.notifications.notifications.logger"), patch(
            "backend.notifications.notifications.set_user_email_verification",
            new_callable=AsyncMock,
        ) as mock_set_verification, patch(
            "backend.notifications.notifications.disable_all_user_notifications",
            new_callable=AsyncMock,
        ) as mock_disable_all, patch(
            "backend.notifications.notifications.get_database_manager_async_client"
        ) as mock_db_client, patch(
            "backend.notifications.notifications.generate_unsubscribe_link"
        ) as mock_unsub_link:

            # Create batch of 5 notifications
            notifications = []
            for i in range(5):
                notification = Mock()
                notification.id = f"notif_{i}"
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
            mock_db.clear_all_user_notification_batches = AsyncMock()
            mock_db.remove_notifications_from_batch = AsyncMock()
            mock_unsub_link.return_value = "http://example.com/unsub"

            # Mock internal methods
            notification_manager._should_email_user_based_on_preference = AsyncMock(
                return_value=True
            )
            notification_manager._should_batch = AsyncMock(return_value=True)
            notification_manager._parse_message = Mock(return_value=sample_batch_event)

            # Track calls
            call_count = [0]

            def send_side_effect(*args, **kwargs):
                data = kwargs.get("data", [])
                if isinstance(data, list) and len(data) == 1:
                    current_call = call_count[0]
                    call_count[0] += 1

                    # First two succeed, third hits 406
                    if current_call < 2:
                        return None
                    else:
                        raise Exception("Recipient marked as inactive (406)")
                # Force single processing
                raise Exception("Force single processing")

            notification_manager.email_sender.send_templated.side_effect = (
                send_side_effect
            )

            # Act
            result = await notification_manager._process_batch(
                sample_batch_event.model_dump_json()
            )

            # Assert
            assert result is True

            # Only 3 calls should have been made (2 successful, 1 failed with 406)
            assert call_count[0] == 3

            # User should be deactivated
            mock_set_verification.assert_called_once_with("user_1", False)
            mock_disable_all.assert_called_once_with("user_1")
            mock_db.clear_all_user_notification_batches.assert_called_once_with(
                "user_1"
            )

            # No further processing should occur after 406

    @pytest.mark.asyncio
    async def test_422_permanently_removes_malformed_notification(
        self, notification_manager, sample_batch_event
    ):
        """Test that 422 error permanently removes the malformed notification from batch and continues with others."""
        with patch("backend.notifications.notifications.logger") as mock_logger, patch(
            "backend.notifications.notifications.get_database_manager_async_client"
        ) as mock_db_client, patch(
            "backend.notifications.notifications.generate_unsubscribe_link"
        ) as mock_unsub_link:

            # Create batch of 5 notifications
            notifications = []
            for i in range(5):
                notification = Mock()
                notification.id = f"notif_{i}"
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
                side_effect=[
                    Mock(notifications=notifications),
                    Mock(notifications=[]),  # Empty after processing
                ]
            )
            mock_db.remove_notifications_from_batch = AsyncMock()
            mock_unsub_link.return_value = "http://example.com/unsub"

            # Mock internal methods
            notification_manager._should_email_user_based_on_preference = AsyncMock(
                return_value=True
            )
            notification_manager._should_batch = AsyncMock(return_value=True)
            notification_manager._parse_message = Mock(return_value=sample_batch_event)

            # Track calls
            call_count = [0]
            successful_indices = []
            removed_notification_ids = []

            # Capture what gets removed
            def remove_side_effect(user_id, notif_type, notif_ids):
                removed_notification_ids.extend(notif_ids)
                return None

            mock_db.remove_notifications_from_batch.side_effect = remove_side_effect

            def send_side_effect(*args, **kwargs):
                data = kwargs.get("data", [])
                if isinstance(data, list) and len(data) == 1:
                    current_call = call_count[0]
                    call_count[0] += 1

                    # Index 2 has malformed data (422)
                    if current_call == 2:
                        raise Exception(
                            "Unprocessable entity (422): Malformed email data"
                        )
                    else:
                        successful_indices.append(current_call)
                        return None
                # Force single processing
                raise Exception("Force single processing")

            notification_manager.email_sender.send_templated.side_effect = (
                send_side_effect
            )

            # Act
            result = await notification_manager._process_batch(
                sample_batch_event.model_dump_json()
            )

            # Assert
            assert result is True
            assert call_count[0] == 5  # All 5 attempted
            assert len(successful_indices) == 4  # 4 succeeded (all except index 2)
            assert 2 not in successful_indices  # Index 2 failed

            # Verify 422 error was logged
            error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
            assert any(
                "422" in call or "malformed" in call.lower() for call in error_calls
            )

            # Verify all notifications were removed (4 successful + 1 malformed)
            assert mock_db.remove_notifications_from_batch.call_count == 5
            assert (
                "notif_2" in removed_notification_ids
            )  # Malformed one was removed permanently

    @pytest.mark.asyncio
    async def test_oversized_notification_permanently_removed(
        self, notification_manager, sample_batch_event
    ):
        """Test that oversized notifications are permanently removed from batch but others continue."""
        with patch("backend.notifications.notifications.logger") as mock_logger, patch(
            "backend.notifications.notifications.get_database_manager_async_client"
        ) as mock_db_client, patch(
            "backend.notifications.notifications.generate_unsubscribe_link"
        ) as mock_unsub_link:

            # Create batch of 5 notifications
            notifications = []
            for i in range(5):
                notification = Mock()
                notification.id = f"notif_{i}"
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
                side_effect=[
                    Mock(notifications=notifications),
                    Mock(notifications=[]),  # Empty after processing
                ]
            )
            mock_db.remove_notifications_from_batch = AsyncMock()
            mock_unsub_link.return_value = "http://example.com/unsub"

            # Mock internal methods
            notification_manager._should_email_user_based_on_preference = AsyncMock(
                return_value=True
            )
            notification_manager._should_batch = AsyncMock(return_value=True)
            notification_manager._parse_message = Mock(return_value=sample_batch_event)

            # Override formatter to simulate oversized on index 3
            # original_format = notification_manager.email_sender.formatter.format_email

            def format_side_effect(*args, **kwargs):
                # Check if we're formatting index 3
                data = kwargs.get("data", {}).get("notifications", [])
                if data and len(data) == 1:
                    # Check notification content to identify index 3
                    if any(
                        "Test Agent 3" in str(n.data)
                        for n in data
                        if hasattr(n, "data")
                    ):
                        # Return oversized message for index 3
                        return ("subject", "x" * 5_000_000)  # Over 4.5MB limit
                return ("subject", "normal sized content")

            notification_manager.email_sender.formatter.format_email = Mock(
                side_effect=format_side_effect
            )

            # Track calls
            successful_indices = []

            def send_side_effect(*args, **kwargs):
                data = kwargs.get("data", [])
                if isinstance(data, list) and len(data) == 1:
                    # Track which notification was sent based on content
                    for i, notif in enumerate(notifications):
                        if any(
                            f"Test Agent {i}" in str(n.data)
                            for n in data
                            if hasattr(n, "data")
                        ):
                            successful_indices.append(i)
                            return None
                    return None
                # Force single processing
                raise Exception("Force single processing")

            notification_manager.email_sender.send_templated.side_effect = (
                send_side_effect
            )

            # Act
            result = await notification_manager._process_batch(
                sample_batch_event.model_dump_json()
            )

            # Assert
            assert result is True
            assert (
                len(successful_indices) == 4
            )  # Only 4 sent (index 3 skipped due to size)
            assert 3 not in successful_indices  # Index 3 was not sent

            # Verify oversized error was logged
            error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
            assert any(
                "exceeds email size limit" in call or "oversized" in call.lower()
                for call in error_calls
            )

    @pytest.mark.asyncio
    async def test_generic_api_error_keeps_notification_for_retry(
        self, notification_manager, sample_batch_event
    ):
        """Test that generic API errors keep notifications in batch for retry while others continue."""
        with patch("backend.notifications.notifications.logger") as mock_logger, patch(
            "backend.notifications.notifications.get_database_manager_async_client"
        ) as mock_db_client, patch(
            "backend.notifications.notifications.generate_unsubscribe_link"
        ) as mock_unsub_link:

            # Create batch of 5 notifications
            notifications = []
            for i in range(5):
                notification = Mock()
                notification.id = f"notif_{i}"
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

            # Notification that failed with generic error
            failed_notifications = [notifications[1]]  # Only index 1 remains for retry

            # Setup mocks
            mock_db = mock_db_client.return_value
            mock_db.get_user_email_by_id = AsyncMock(return_value="test@example.com")
            mock_db.get_user_notification_batch = AsyncMock(
                side_effect=[
                    Mock(notifications=notifications),
                    Mock(
                        notifications=failed_notifications
                    ),  # Failed ones remain for retry
                ]
            )
            mock_db.remove_notifications_from_batch = AsyncMock()
            mock_unsub_link.return_value = "http://example.com/unsub"

            # Mock internal methods
            notification_manager._should_email_user_based_on_preference = AsyncMock(
                return_value=True
            )
            notification_manager._should_batch = AsyncMock(return_value=True)
            notification_manager._parse_message = Mock(return_value=sample_batch_event)

            # Track calls
            successful_indices = []
            failed_indices = []
            removed_notification_ids = []

            # Capture what gets removed
            def remove_side_effect(user_id, notif_type, notif_ids):
                removed_notification_ids.extend(notif_ids)
                return None

            mock_db.remove_notifications_from_batch.side_effect = remove_side_effect

            def send_side_effect(*args, **kwargs):
                data = kwargs.get("data", [])
                if isinstance(data, list) and len(data) == 1:
                    # Track which notification based on content
                    for i, notif in enumerate(notifications):
                        if any(
                            f"Test Agent {i}" in str(n.data)
                            for n in data
                            if hasattr(n, "data")
                        ):
                            # Index 1 has generic API error
                            if i == 1:
                                failed_indices.append(i)
                                raise Exception("Network timeout - temporary failure")
                            else:
                                successful_indices.append(i)
                                return None
                    return None
                # Force single processing
                raise Exception("Force single processing")

            notification_manager.email_sender.send_templated.side_effect = (
                send_side_effect
            )

            # Act
            result = await notification_manager._process_batch(
                sample_batch_event.model_dump_json()
            )

            # Assert
            assert result is True
            assert len(successful_indices) == 4  # 4 succeeded (0, 2, 3, 4)
            assert len(failed_indices) == 1  # 1 failed
            assert 1 in failed_indices  # Index 1 failed

            # Verify generic error was logged
            error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
            assert any(
                "api error" in call.lower() or "skipping" in call.lower()
                for call in error_calls
            )

            # Only successful ones should be removed from batch (failed one stays for retry)
            assert mock_db.remove_notifications_from_batch.call_count == 4
            assert (
                "notif_1" not in removed_notification_ids
            )  # Failed one NOT removed (stays for retry)
            assert "notif_0" in removed_notification_ids  # Successful one removed
            assert "notif_2" in removed_notification_ids  # Successful one removed
            assert "notif_3" in removed_notification_ids  # Successful one removed
            assert "notif_4" in removed_notification_ids  # Successful one removed

    @pytest.mark.asyncio
    async def test_batch_all_notifications_sent_successfully(
        self, notification_manager, sample_batch_event
    ):
        """Test successful batch processing where all notifications are sent without errors."""
        with patch("backend.notifications.notifications.logger") as mock_logger, patch(
            "backend.notifications.notifications.get_database_manager_async_client"
        ) as mock_db_client, patch(
            "backend.notifications.notifications.generate_unsubscribe_link"
        ) as mock_unsub_link:

            # Create batch of 5 notifications
            notifications = []
            for i in range(5):
                notification = Mock()
                notification.id = f"notif_{i}"
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
                side_effect=[
                    Mock(notifications=notifications),
                    Mock(notifications=[]),  # Empty after all sent successfully
                ]
            )
            mock_db.remove_notifications_from_batch = AsyncMock()
            mock_unsub_link.return_value = "http://example.com/unsub"

            # Mock internal methods
            notification_manager._should_email_user_based_on_preference = AsyncMock(
                return_value=True
            )
            notification_manager._should_batch = AsyncMock(return_value=True)
            notification_manager._parse_message = Mock(return_value=sample_batch_event)

            # Track successful sends
            successful_indices = []
            removed_notification_ids = []

            # Capture what gets removed
            def remove_side_effect(user_id, notif_type, notif_ids):
                removed_notification_ids.extend(notif_ids)
                return None

            mock_db.remove_notifications_from_batch.side_effect = remove_side_effect

            def send_side_effect(*args, **kwargs):
                data = kwargs.get("data", [])
                if isinstance(data, list) and len(data) == 1:
                    # Track which notification was sent
                    for i, notif in enumerate(notifications):
                        if any(
                            f"Test Agent {i}" in str(n.data)
                            for n in data
                            if hasattr(n, "data")
                        ):
                            successful_indices.append(i)
                            return None
                    return None  # Success
                # Force single processing
                raise Exception("Force single processing")

            notification_manager.email_sender.send_templated.side_effect = (
                send_side_effect
            )

            # Act
            result = await notification_manager._process_batch(
                sample_batch_event.model_dump_json()
            )

            # Assert
            assert result is True

            # All 5 notifications should be sent successfully
            assert len(successful_indices) == 5
            assert successful_indices == [0, 1, 2, 3, 4]

            # All notifications should be removed from batch
            assert mock_db.remove_notifications_from_batch.call_count == 5
            assert len(removed_notification_ids) == 5
            for i in range(5):
                assert f"notif_{i}" in removed_notification_ids

            # No errors should be logged
            assert mock_logger.error.call_count == 0

            # Info message about successful sends should be logged
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("sent and removed" in call.lower() for call in info_calls)
