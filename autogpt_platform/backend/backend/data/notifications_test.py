"""Tests for notification data models and batch helpers."""

import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from prisma.actions import UserNotificationBatchActions
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
    empty_user_notification_batch,
)
from backend.util.test import SpinTestServer

logger = logging.getLogger(__name__)


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


def _make_agent_run_event(user_id: str, agent_name: str = "Test Agent"):
    return NotificationEventModel[AgentRunData](
        type=NotificationType.AGENT_RUN,
        user_id=user_id,
        data=AgentRunData(
            agent_name=agent_name,
            credits_used=10.0,
            execution_time=5.0,
            node_count=3,
            graph_id="graph-" + agent_name,
            outputs=[],
        ),
    )


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
        await empty_user_notification_batch(user_id, NotificationType.AGENT_RUN)
    except Exception as exc:
        logger.warning("notification cleanup for %s failed: %s", user_id, exc)
    try:
        await User.prisma().delete_many(where={"id": user_id})
    except Exception as exc:
        logger.warning("user cleanup for %s failed: %s", user_id, exc)


@pytest.mark.asyncio(loop_scope="session")
async def test_upsert_creates_empty_batch_then_appends(server: SpinTestServer):
    """Empty batch path creates a batch. Existing-batch path appends without
    loading the existing notification rows."""
    user_id = f"notif-create-{uuid4()}"
    await _create_test_user(user_id)

    try:
        # First call → create path.
        first = await create_or_add_to_user_notification_batch(
            user_id=user_id,
            notification_type=NotificationType.AGENT_RUN,
            notification_data=_make_agent_run_event(user_id, agent_name="first"),
        )
        assert first.user_id == user_id
        assert first.type == NotificationType.AGENT_RUN
        # The returned DTO no longer eagerly loads notifications — callers
        # that need them must fetch separately.
        assert first.notifications == []

        row_count = await NotificationEvent.prisma().count(
            where={"UserNotificationBatch": {"is": {"userId": user_id}}}
        )
        assert row_count == 1

        # Second call → update path.
        second = await create_or_add_to_user_notification_batch(
            user_id=user_id,
            notification_type=NotificationType.AGENT_RUN,
            notification_data=_make_agent_run_event(user_id, agent_name="second"),
        )
        assert second.user_id == user_id
        # Still no eager include, even when the batch has rows.
        assert second.notifications == []

        row_count = await NotificationEvent.prisma().count(
            where={"UserNotificationBatch": {"is": {"userId": user_id}}}
        )
        assert row_count == 2

        # Exactly one batch for (user_id, AGENT_RUN).
        batch_count = await UserNotificationBatch.prisma().count(
            where={"userId": user_id, "type": NotificationType.AGENT_RUN}
        )
        assert batch_count == 1
    finally:
        await _cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_upsert_does_not_load_existing_notifications(server: SpinTestServer):
    """Pre-seeding 25 notifications into the batch and then upserting one more
    must not trip on the eager include that used to load every row."""
    user_id = f"notif-bigbatch-{uuid4()}"
    await _create_test_user(user_id)

    try:
        # Seed the batch by issuing 25 upserts. The first creates, the rest
        # append.
        for i in range(25):
            await create_or_add_to_user_notification_batch(
                user_id=user_id,
                notification_type=NotificationType.AGENT_RUN,
                notification_data=_make_agent_run_event(user_id, agent_name=f"a{i}"),
            )

        pre_count = await NotificationEvent.prisma().count(
            where={"UserNotificationBatch": {"is": {"userId": user_id}}}
        )
        assert pre_count == 25

        # The 26th upsert: must succeed and must NOT return the 25 prior rows.
        resp = await create_or_add_to_user_notification_batch(
            user_id=user_id,
            notification_type=NotificationType.AGENT_RUN,
            notification_data=_make_agent_run_event(user_id, agent_name="final"),
        )
        assert resp.notifications == [], (
            "Upsert must not eagerly include Notifications; that is the "
            "statement_timeout regression we are guarding against."
        )

        post_count = await NotificationEvent.prisma().count(
            where={"UserNotificationBatch": {"is": {"userId": user_id}}}
        )
        assert post_count == 26
    finally:
        await _cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_upsert_concurrent_invocations_no_unique_violation(
    server: SpinTestServer,
):
    """Two concurrent invocations on an empty batch must both succeed: one
    creates, one appends. Neither loses its event, and no
    @@unique([userId, type]) violation leaks out of the helper.

    Prisma's upsert is find→INSERT/UPDATE, not a true SQL ON CONFLICT, so
    two concurrent calls can both miss the existing row and both attempt
    INSERT. The helper retries on UniqueViolationError so the loser
    converges on the UPDATE path."""
    user_id = f"notif-race-{uuid4()}"
    await _create_test_user(user_id)

    try:
        results = await asyncio.gather(
            create_or_add_to_user_notification_batch(
                user_id=user_id,
                notification_type=NotificationType.AGENT_RUN,
                notification_data=_make_agent_run_event(user_id, agent_name="left"),
            ),
            create_or_add_to_user_notification_batch(
                user_id=user_id,
                notification_type=NotificationType.AGENT_RUN,
                notification_data=_make_agent_run_event(user_id, agent_name="right"),
            ),
            return_exceptions=True,
        )

        assert all(not isinstance(r, Exception) for r in results), results

        batch_count = await UserNotificationBatch.prisma().count(
            where={"userId": user_id, "type": NotificationType.AGENT_RUN}
        )
        assert batch_count == 1

        event_count = await NotificationEvent.prisma().count(
            where={"UserNotificationBatch": {"is": {"userId": user_id}}}
        )
        assert event_count == 2, "both concurrent notifications must persist"
    finally:
        await _cleanup_test_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_upsert_retries_on_unique_violation(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    """Deterministic coverage for the retry path: when the first upsert
    raises UniqueViolationError (the loser of a concurrent INSERT race),
    the helper retries and the second attempt succeeds via UPDATE.

    Async gather + a shared connection pool can serialize concurrent calls
    in some test setups, so we patch the actions class to inject a unique
    violation on the first call and prove the retry actually runs."""
    user_id = f"notif-retry-{uuid4()}"
    await _create_test_user(user_id)

    try:
        # Pre-create the batch so the retry's UPDATE path has a row to hit.
        await create_or_add_to_user_notification_batch(
            user_id=user_id,
            notification_type=NotificationType.AGENT_RUN,
            notification_data=_make_agent_run_event(user_id, agent_name="seed"),
        )

        original_upsert = UserNotificationBatchActions.upsert
        call_count = {"n": 0}

        async def flaky_upsert(self, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise UniqueViolationError(
                    {
                        "error": (
                            "Unique constraint failed on the fields: (`userId`,`type`)"
                        ),
                        "user_facing_error": {},
                    }
                )
            return await original_upsert(self, *args, **kwargs)

        monkeypatch.setattr(UserNotificationBatchActions, "upsert", flaky_upsert)

        resp = await create_or_add_to_user_notification_batch(
            user_id=user_id,
            notification_type=NotificationType.AGENT_RUN,
            notification_data=_make_agent_run_event(user_id, agent_name="retry"),
        )
        assert resp.user_id == user_id
        assert call_count["n"] == 2, "retry path must run exactly once"

        event_count = await NotificationEvent.prisma().count(
            where={"UserNotificationBatch": {"is": {"userId": user_id}}}
        )
        assert event_count == 2
    finally:
        await _cleanup_test_user(user_id)
