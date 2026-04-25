"""Tests for notification data models."""

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from prisma.enums import NotificationType
from prisma.errors import UniqueViolationError
from prisma.models import NotificationEvent, User, UserNotificationBatch
from pydantic import ValidationError

from backend.data.notifications import (
    AgentApprovalData,
    AgentRejectionData,
    AgentRunData,
    NotificationEventModel,
    create_or_add_to_user_notification_batch,
)
from backend.util.test import SpinTestServer


class TestAgentApprovalData:
    """Test cases for AgentApprovalData model."""

    def test_valid_agent_approval_data(self):
        """Test creating valid AgentApprovalData."""
        data = AgentApprovalData(
            agent_name="Test Agent",
            graph_id="test-agent-123",
            graph_version=1,
            reviewer_name="John Doe",
            reviewer_email="john@example.com",
            comments="Great agent, approved!",
            reviewed_at=datetime.now(timezone.utc),
            store_url="https://app.autogpt.com/store/test-agent-123",
        )

        assert data.agent_name == "Test Agent"
        assert data.graph_id == "test-agent-123"
        assert data.graph_version == 1
        assert data.reviewer_name == "John Doe"
        assert data.reviewer_email == "john@example.com"
        assert data.comments == "Great agent, approved!"
        assert data.store_url == "https://app.autogpt.com/store/test-agent-123"
        assert data.reviewed_at.tzinfo is not None

    def test_agent_approval_data_without_timezone_raises_error(self):
        """Test that AgentApprovalData raises error without timezone."""
        with pytest.raises(
            ValidationError, match="datetime must have timezone information"
        ):
            AgentApprovalData(
                agent_name="Test Agent",
                graph_id="test-agent-123",
                graph_version=1,
                reviewer_name="John Doe",
                reviewer_email="john@example.com",
                comments="Great agent, approved!",
                reviewed_at=datetime.now(),  # No timezone
                store_url="https://app.autogpt.com/store/test-agent-123",
            )

    def test_agent_approval_data_with_empty_comments(self):
        """Test AgentApprovalData with empty comments."""
        data = AgentApprovalData(
            agent_name="Test Agent",
            graph_id="test-agent-123",
            graph_version=1,
            reviewer_name="John Doe",
            reviewer_email="john@example.com",
            comments="",  # Empty comments
            reviewed_at=datetime.now(timezone.utc),
            store_url="https://app.autogpt.com/store/test-agent-123",
        )

        assert data.comments == ""


class TestAgentRejectionData:
    """Test cases for AgentRejectionData model."""

    def test_valid_agent_rejection_data(self):
        """Test creating valid AgentRejectionData."""
        data = AgentRejectionData(
            agent_name="Test Agent",
            graph_id="test-agent-123",
            graph_version=1,
            reviewer_name="Jane Doe",
            reviewer_email="jane@example.com",
            comments="Please fix the security issues before resubmitting.",
            reviewed_at=datetime.now(timezone.utc),
            resubmit_url="https://app.autogpt.com/build/test-agent-123",
        )

        assert data.agent_name == "Test Agent"
        assert data.graph_id == "test-agent-123"
        assert data.graph_version == 1
        assert data.reviewer_name == "Jane Doe"
        assert data.reviewer_email == "jane@example.com"
        assert data.comments == "Please fix the security issues before resubmitting."
        assert data.resubmit_url == "https://app.autogpt.com/build/test-agent-123"
        assert data.reviewed_at.tzinfo is not None

    def test_agent_rejection_data_without_timezone_raises_error(self):
        """Test that AgentRejectionData raises error without timezone."""
        with pytest.raises(
            ValidationError, match="datetime must have timezone information"
        ):
            AgentRejectionData(
                agent_name="Test Agent",
                graph_id="test-agent-123",
                graph_version=1,
                reviewer_name="Jane Doe",
                reviewer_email="jane@example.com",
                comments="Please fix the security issues.",
                reviewed_at=datetime.now(),  # No timezone
                resubmit_url="https://app.autogpt.com/build/test-agent-123",
            )

    def test_agent_rejection_data_with_long_comments(self):
        """Test AgentRejectionData with long comments."""
        long_comment = "A" * 1000  # Very long comment
        data = AgentRejectionData(
            agent_name="Test Agent",
            graph_id="test-agent-123",
            graph_version=1,
            reviewer_name="Jane Doe",
            reviewer_email="jane@example.com",
            comments=long_comment,
            reviewed_at=datetime.now(timezone.utc),
            resubmit_url="https://app.autogpt.com/build/test-agent-123",
        )

        assert data.comments == long_comment

    def test_model_serialization(self):
        """Test that models can be serialized and deserialized."""
        original_data = AgentRejectionData(
            agent_name="Test Agent",
            graph_id="test-agent-123",
            graph_version=1,
            reviewer_name="Jane Doe",
            reviewer_email="jane@example.com",
            comments="Please fix the issues.",
            reviewed_at=datetime.now(timezone.utc),
            resubmit_url="https://app.autogpt.com/build/test-agent-123",
        )

        # Serialize to dict
        data_dict = original_data.model_dump()

        # Deserialize back
        restored_data = AgentRejectionData.model_validate(data_dict)

        assert restored_data.agent_name == original_data.agent_name
        assert restored_data.graph_id == original_data.graph_id
        assert restored_data.graph_version == original_data.graph_version
        assert restored_data.reviewer_name == original_data.reviewer_name
        assert restored_data.reviewer_email == original_data.reviewer_email
        assert restored_data.comments == original_data.comments
        assert restored_data.reviewed_at == original_data.reviewed_at
        assert restored_data.resubmit_url == original_data.resubmit_url


# ---------- create_or_add_to_user_notification_batch ----------


async def _create_test_user(user_id: str) -> None:
    try:
        await User.prisma().create(
            data={
                "id": user_id,
                "email": f"test-{user_id}@example.com",
                "name": f"Test User {user_id[:8]}",
            }
        )
    except UniqueViolationError:
        pass


async def _cleanup_test_user(user_id: str) -> None:
    try:
        batches = await UserNotificationBatch.prisma().find_many(
            where={"userId": user_id}
        )
        for batch in batches:
            await NotificationEvent.prisma().delete_many(
                where={"userNotificationBatchId": batch.id}
            )
        await UserNotificationBatch.prisma().delete_many(where={"userId": user_id})
        await User.prisma().delete_many(where={"id": user_id})
    except Exception as e:  # noqa: BLE001 - cleanup is best-effort
        print(f"cleanup for {user_id}: {e}")


def _make_event(user_id: str) -> NotificationEventModel[AgentRunData]:
    return NotificationEventModel[AgentRunData](
        type=NotificationType.AGENT_RUN,
        user_id=user_id,
        data=AgentRunData(
            agent_name="Test Agent",
            credits_used=1.0,
            execution_time=0.1,
            node_count=1,
            graph_id="test-graph",
            outputs=[{"k": "v"}],
        ),
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_batch_upsert_creates_first_notification(server: SpinTestServer):
    """First invocation with no existing batch creates it with the event."""
    user_id = f"batch-new-{uuid4()}"
    await _create_test_user(user_id)
    try:
        dto = await create_or_add_to_user_notification_batch(
            user_id=user_id,
            notification_type=NotificationType.AGENT_RUN,
            notification_data=_make_event(user_id),
        )
        assert dto.user_id == user_id
        assert dto.type == NotificationType.AGENT_RUN

        # Verify exactly one NotificationEvent was created on the DB.
        events = await NotificationEvent.prisma().find_many(
            where={
                "UserNotificationBatch": {
                    "is": {"userId": user_id, "type": NotificationType.AGENT_RUN}
                }
            }
        )
        assert len(events) == 1
    finally:
        await _cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_batch_upsert_appends_without_eager_loading(server: SpinTestServer):
    """Existing batch with many events: upsert adds 1, DTO isn't bloated."""
    user_id = f"batch-append-{uuid4()}"
    await _create_test_user(user_id)
    try:
        # Seed a batch with 10 pre-existing events. The real-world heavy case
        # has thousands; 10 is enough to prove we don't rely on eager loading.
        seed_count = 10
        for _ in range(seed_count):
            await create_or_add_to_user_notification_batch(
                user_id=user_id,
                notification_type=NotificationType.AGENT_RUN,
                notification_data=_make_event(user_id),
            )

        dto = await create_or_add_to_user_notification_batch(
            user_id=user_id,
            notification_type=NotificationType.AGENT_RUN,
            notification_data=_make_event(user_id),
        )

        # DTO should NOT eagerly include the full events list.
        assert dto.notifications == []

        # Exactly one batch exists with seed_count + 1 events.
        batches = await UserNotificationBatch.prisma().find_many(
            where={"userId": user_id, "type": NotificationType.AGENT_RUN}
        )
        assert len(batches) == 1

        events = await NotificationEvent.prisma().find_many(
            where={"userNotificationBatchId": batches[0].id}
        )
        assert len(events) == seed_count + 1
    finally:
        await _cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_batch_upsert_concurrent_no_duplicates(server: SpinTestServer):
    """N concurrent upserts for a new (user, type): one batch, N events, no races."""
    user_id = f"batch-race-{uuid4()}"
    await _create_test_user(user_id)
    try:
        n = 10
        results = await asyncio.gather(
            *[
                create_or_add_to_user_notification_batch(
                    user_id=user_id,
                    notification_type=NotificationType.AGENT_RUN,
                    notification_data=_make_event(user_id),
                )
                for _ in range(n)
            ],
            return_exceptions=True,
        )
        failures = [r for r in results if isinstance(r, Exception)]
        assert not failures, f"upsert race surfaced exceptions: {failures}"

        # Exactly one batch — the @@unique([userId, type]) index + upsert must
        # collapse concurrent writers onto the same row.
        batches = await UserNotificationBatch.prisma().find_many(
            where={"userId": user_id, "type": NotificationType.AGENT_RUN}
        )
        assert len(batches) == 1, (
            f"expected exactly 1 batch, got {len(batches)}: {batches}"
        )

        # Every notification landed — no dropped writes.
        events = await NotificationEvent.prisma().find_many(
            where={"userNotificationBatchId": batches[0].id}
        )
        assert len(events) == n, (
            f"expected {n} events, got {len(events)} — writes were lost"
        )
    finally:
        await _cleanup_test_user(user_id)
