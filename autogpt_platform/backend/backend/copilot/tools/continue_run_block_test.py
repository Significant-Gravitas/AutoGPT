"""Tests for ContinueRunBlockTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ReviewStatus

from ._test_data import make_session
from .continue_run_block import ContinueRunBlockTool
from .models import BlockOutputResponse, ErrorResponse

_TEST_USER_ID = "test-user-continue"


def _make_review(
    node_exec_id: str,
    user_id: str,
    status: ReviewStatus = ReviewStatus.APPROVED,
    payload: dict | None = None,
):
    """Create a mock PendingHumanReview record."""
    mock = MagicMock()
    mock.nodeExecId = node_exec_id
    mock.userId = user_id
    mock.status = status
    mock.payload = payload or {"text": "hello"}
    return mock


class TestContinueRunBlock:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_missing_review_id_returns_error(self):
        tool = ContinueRunBlockTool()
        session = make_session(user_id=_TEST_USER_ID)

        response = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            review_id="",
        )

        assert isinstance(response, ErrorResponse)
        assert "review_id" in response.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_review_not_found_returns_error(self):
        tool = ContinueRunBlockTool()
        session = make_session(user_id=_TEST_USER_ID)

        with patch(
            "backend.copilot.tools.continue_run_block.PendingHumanReview"
        ) as mock_model:
            mock_model.prisma.return_value.find_first = AsyncMock(return_value=None)

            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                review_id="copilot-node-some-block:abc12345",
            )

        assert isinstance(response, ErrorResponse)
        assert "not found" in response.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_waiting_review_returns_error(self):
        tool = ContinueRunBlockTool()
        session = make_session(user_id=_TEST_USER_ID)
        review = _make_review(
            "copilot-node-some-block:abc12345",
            _TEST_USER_ID,
            status=ReviewStatus.WAITING,
        )

        with patch(
            "backend.copilot.tools.continue_run_block.PendingHumanReview"
        ) as mock_model:
            mock_model.prisma.return_value.find_first = AsyncMock(return_value=review)

            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                review_id="copilot-node-some-block:abc12345",
            )

        assert isinstance(response, ErrorResponse)
        assert "not been approved" in response.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_rejected_review_returns_error(self):
        tool = ContinueRunBlockTool()
        session = make_session(user_id=_TEST_USER_ID)
        review = _make_review(
            "copilot-node-some-block:abc12345",
            _TEST_USER_ID,
            status=ReviewStatus.REJECTED,
        )

        with patch(
            "backend.copilot.tools.continue_run_block.PendingHumanReview"
        ) as mock_model:
            mock_model.prisma.return_value.find_first = AsyncMock(return_value=review)

            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                review_id="copilot-node-some-block:abc12345",
            )

        assert isinstance(response, ErrorResponse)
        assert "rejected" in response.message.lower()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_approved_review_executes_block(self):
        tool = ContinueRunBlockTool()
        session = make_session(user_id=_TEST_USER_ID)
        review_id = "copilot-node-delete-branch-id:abc12345"
        input_data = {"repo_url": "https://github.com/test/repo", "branch": "main"}
        review = _make_review(
            review_id,
            _TEST_USER_ID,
            status=ReviewStatus.APPROVED,
            payload=input_data,
        )

        mock_block = MagicMock()
        mock_block.name = "Delete Branch"

        async def mock_execute(data, **kwargs):
            yield "result", "Branch deleted"

        mock_block.execute = mock_execute
        mock_block.input_schema.get_credentials_fields_info.return_value = []

        mock_workspace_db = MagicMock()
        mock_workspace_db.get_or_create_workspace = AsyncMock(
            return_value=MagicMock(id="test-workspace-id")
        )

        with (
            patch(
                "backend.copilot.tools.continue_run_block.PendingHumanReview"
            ) as mock_model,
            patch(
                "backend.copilot.tools.continue_run_block.get_block",
                return_value=mock_block,
            ),
            patch(
                "backend.copilot.tools.continue_run_block.workspace_db",
                return_value=mock_workspace_db,
            ),
        ):
            mock_model.prisma.return_value.find_first = AsyncMock(return_value=review)
            mock_model.prisma.return_value.delete_many = AsyncMock()

            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                review_id=review_id,
            )

        assert isinstance(response, BlockOutputResponse)
        assert response.success is True
        assert response.block_name == "Delete Branch"
        # Verify review was deleted (one-time use)
        mock_model.prisma.return_value.delete_many.assert_called_once()
