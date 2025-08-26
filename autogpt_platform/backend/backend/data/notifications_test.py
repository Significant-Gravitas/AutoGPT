"""Tests for notification data models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.data.notifications import AgentApprovalData, AgentRejectionData


class TestAgentApprovalData:
    """Test cases for AgentApprovalData model."""

    def test_valid_agent_approval_data(self):
        """Test creating valid AgentApprovalData."""
        data = AgentApprovalData(
            agent_name="Test Agent",
            agent_id="test-agent-123",
            agent_version=1,
            reviewer_name="John Doe",
            reviewer_email="john@example.com",
            comments="Great agent, approved!",
            reviewed_at=datetime.now(timezone.utc),
            store_url="https://app.autogpt.com/store/test-agent-123",
        )

        assert data.agent_name == "Test Agent"
        assert data.agent_id == "test-agent-123"
        assert data.agent_version == 1
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
                agent_id="test-agent-123",
                agent_version=1,
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
            agent_id="test-agent-123",
            agent_version=1,
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
            agent_id="test-agent-123",
            agent_version=1,
            reviewer_name="Jane Doe",
            reviewer_email="jane@example.com",
            comments="Please fix the security issues before resubmitting.",
            reviewed_at=datetime.now(timezone.utc),
            resubmit_url="https://app.autogpt.com/build/test-agent-123",
        )

        assert data.agent_name == "Test Agent"
        assert data.agent_id == "test-agent-123"
        assert data.agent_version == 1
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
                agent_id="test-agent-123",
                agent_version=1,
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
            agent_id="test-agent-123",
            agent_version=1,
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
            agent_id="test-agent-123",
            agent_version=1,
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
        assert restored_data.agent_id == original_data.agent_id
        assert restored_data.agent_version == original_data.agent_version
        assert restored_data.reviewer_name == original_data.reviewer_name
        assert restored_data.reviewer_email == original_data.reviewer_email
        assert restored_data.comments == original_data.comments
        assert restored_data.reviewed_at == original_data.reviewed_at
        assert restored_data.resubmit_url == original_data.resubmit_url
