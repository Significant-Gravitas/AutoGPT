"""Tests for ContinueRunBlockTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ReviewStatus

from ._test_data import make_session
from .continue_run_block import ContinueRunBlockTool
from .models import BlockOutputResponse, ErrorResponse

_TEST_USER_ID = "test-user-continue"


def _make_review_model(
    node_exec_id: str,
    status: ReviewStatus = ReviewStatus.APPROVED,
    payload: dict | None = None,
    graph_exec_id: str = "",
):
    """Create a mock PendingHumanReviewModel."""
    mock = MagicMock()
    mock.node_exec_id = node_exec_id
    mock.status = status
    mock.payload = payload or {"text": "hello"}
    mock.graph_exec_id = graph_exec_id
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

        mock_db = MagicMock()
        mock_db.get_reviews_by_node_exec_ids = AsyncMock(return_value={})

        with patch(
            "backend.copilot.tools.continue_run_block.review_db",
            return_value=mock_db,
        ):
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
        review_id = "copilot-node-some-block:abc12345"
        graph_exec_id = f"copilot-session-{session.session_id}"
        review = _make_review_model(
            review_id, status=ReviewStatus.WAITING, graph_exec_id=graph_exec_id
        )

        mock_db = MagicMock()
        mock_db.get_reviews_by_node_exec_ids = AsyncMock(
            return_value={review_id: review}
        )

        with patch(
            "backend.copilot.tools.continue_run_block.review_db",
            return_value=mock_db,
        ):
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                review_id=review_id,
            )

        assert isinstance(response, ErrorResponse)
        assert "not been approved" in response.message

    @pytest.mark.asyncio(loop_scope="session")
    async def test_rejected_review_returns_error(self):
        tool = ContinueRunBlockTool()
        session = make_session(user_id=_TEST_USER_ID)
        review_id = "copilot-node-some-block:abc12345"
        graph_exec_id = f"copilot-session-{session.session_id}"
        review = _make_review_model(
            review_id, status=ReviewStatus.REJECTED, graph_exec_id=graph_exec_id
        )

        mock_db = MagicMock()
        mock_db.get_reviews_by_node_exec_ids = AsyncMock(
            return_value={review_id: review}
        )

        with patch(
            "backend.copilot.tools.continue_run_block.review_db",
            return_value=mock_db,
        ):
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                review_id=review_id,
            )

        assert isinstance(response, ErrorResponse)
        assert "rejected" in response.message.lower()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_approved_review_executes_block(self):
        tool = ContinueRunBlockTool()
        session = make_session(user_id=_TEST_USER_ID)
        review_id = "copilot-node-delete-branch-id:abc12345"
        graph_exec_id = f"copilot-session-{session.session_id}"
        input_data = {"repo_url": "https://github.com/test/repo", "branch": "main"}
        review = _make_review_model(
            review_id,
            status=ReviewStatus.APPROVED,
            payload=input_data,
            graph_exec_id=graph_exec_id,
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

        mock_db = MagicMock()
        mock_db.get_reviews_by_node_exec_ids = AsyncMock(
            return_value={review_id: review}
        )
        mock_db.delete_review_by_node_exec_id = AsyncMock(return_value=1)

        with (
            patch(
                "backend.copilot.tools.continue_run_block.review_db",
                return_value=mock_db,
            ),
            patch(
                "backend.copilot.tools.continue_run_block.get_block",
                return_value=mock_block,
            ),
            patch(
                "backend.copilot.tools.helpers.workspace_db",
                return_value=mock_workspace_db,
            ),
            patch(
                "backend.copilot.tools.helpers.match_credentials_to_requirements",
                return_value=({}, []),
            ),
        ):
            response = await tool._execute(
                user_id=_TEST_USER_ID,
                session=session,
                review_id=review_id,
            )

        assert isinstance(response, BlockOutputResponse)
        assert response.success is True
        assert response.block_name == "Delete Branch"
        # Verify review was deleted (one-time use)
        mock_db.delete_review_by_node_exec_id.assert_called_once_with(
            review_id, _TEST_USER_ID
        )
