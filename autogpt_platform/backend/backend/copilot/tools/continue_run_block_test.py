"""Tests for ContinueRunBlockTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.blocks._base import BlockType
from backend.copilot.context import _current_permissions
from backend.copilot.permissions import CopilotPermissions
from backend.integrations.providers import ProviderName

from ._test_data import make_session
from .continue_run_block import ContinueRunBlockTool
from .models import BlockOutputResponse, ErrorResponse

_TEST_USER_ID = "test-user-continue"


def _make_review_model(
    node_exec_id: str,
    status: ReviewStatus = ReviewStatus.APPROVED,
    payload: dict | None = None,
    session_id: str = "",
):
    """Create a mock PendingHumanReviewModel."""
    mock = MagicMock()
    mock.node_exec_id = node_exec_id
    mock.status = status
    mock.payload = payload or {"text": "hello"}
    mock.session_id = session_id
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
        review = _make_review_model(
            review_id, status=ReviewStatus.WAITING, session_id=session.session_id
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
        review = _make_review_model(
            review_id, status=ReviewStatus.REJECTED, session_id=session.session_id
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
        input_data = {"repo_url": "https://github.com/test/repo", "branch": "main"}
        review = _make_review_model(
            review_id,
            status=ReviewStatus.APPROVED,
            payload=input_data,
            session_id=session.session_id,
        )

        mock_block = MagicMock()
        mock_block.name = "Delete Branch"
        mock_block.disabled = False

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
        mock_db.claim_approved_review = AsyncMock(return_value=True)
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


@pytest.mark.asyncio(loop_scope="session")
async def test_approved_review_of_a_since_disabled_block_does_not_run():
    """An approval outlives the turn that asked for it, so a block switched off
    in between must still be refused when the approval is used, and the
    approval used up so it cannot run the block once it is re-enabled."""
    tool = ContinueRunBlockTool()
    session = make_session(user_id=_TEST_USER_ID)
    review_id = "copilot-node-some-block:abc12345"
    review = _make_review_model(review_id, session_id=session.session_id)
    mock_db = MagicMock()
    mock_db.get_reviews_by_node_exec_ids = AsyncMock(return_value={review_id: review})
    mock_db.delete_review_by_node_exec_id = AsyncMock(return_value=1)
    block = MagicMock()
    block.name = "Some Block"
    block.disabled = True

    with (
        patch(
            "backend.copilot.tools.continue_run_block.review_db",
            return_value=mock_db,
        ),
        patch("backend.copilot.tools.continue_run_block.get_block", return_value=block),
        patch(
            "backend.copilot.tools.continue_run_block.execute_block",
            new_callable=AsyncMock,
        ) as execute,
    ):
        response = await tool._execute(
            user_id=_TEST_USER_ID, session=session, review_id=review_id
        )

    assert isinstance(response, ErrorResponse)
    assert "disabled" in response.message
    execute.assert_not_awaited()
    mock_db.delete_review_by_node_exec_id.assert_awaited_once_with(
        review_id, _TEST_USER_ID
    )


_BLOCK_ID = "11111111-2222-3333-4444-555555555555"


async def _resume_in_a_turn_with(permissions: CopilotPermissions | None, block):
    """Resume an approved review of *block* in a turn governed by *permissions*."""
    tool = ContinueRunBlockTool()
    session = make_session(user_id=_TEST_USER_ID)
    review_id = f"copilot-node-{_BLOCK_ID}:abc12345"
    review = _make_review_model(review_id, session_id=session.session_id)
    mock_db = MagicMock()
    mock_db.get_reviews_by_node_exec_ids = AsyncMock(return_value={review_id: review})
    mock_db.delete_review_by_node_exec_id = AsyncMock(return_value=1)
    token = _current_permissions.set(permissions)
    try:
        with (
            patch(
                "backend.copilot.tools.continue_run_block.review_db",
                return_value=mock_db,
            ),
            patch(
                "backend.copilot.tools.continue_run_block.get_block",
                return_value=block,
            ),
            patch(
                "backend.copilot.tools.continue_run_block.resolve_block_credentials",
                new_callable=AsyncMock,
                return_value=({}, []),
            ) as resolve,
            patch(
                "backend.copilot.tools.continue_run_block.execute_block",
                new_callable=AsyncMock,
            ) as execute,
        ):
            response = await tool._execute(
                user_id=_TEST_USER_ID, session=session, review_id=review_id
            )
    finally:
        _current_permissions.reset(token)
    return response, resolve, execute, mock_db


def _block(name: str, block_type: BlockType = BlockType.STANDARD) -> MagicMock:
    block = MagicMock()
    block.name = name
    block.disabled = False
    block.block_type = block_type
    return block


@pytest.mark.asyncio(loop_scope="session")
async def test_a_turn_that_does_not_permit_the_block_does_not_run_its_approval():
    """An approval from an earlier turn is no exemption from this turn's
    permissions. The approval stays, for a turn that permits the block."""
    response, resolve, execute, db = await _resume_in_a_turn_with(
        CopilotPermissions(blocks=[_BLOCK_ID], blocks_exclude=True),
        _block("HTTP Request"),
    )

    assert isinstance(response, ErrorResponse)
    assert "not permitted" in response.message
    resolve.assert_not_awaited()
    execute.assert_not_awaited()
    db.delete_review_by_node_exec_id.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_a_turn_whose_provider_ceiling_excludes_the_block_does_not_run_it():
    block = _block("Create PR")
    block.input_schema.get_credentials_fields_info.return_value = {
        "credentials": MagicMock(provider=frozenset({ProviderName.GITHUB}))
    }

    response, resolve, execute, db = await _resume_in_a_turn_with(
        CopilotPermissions(providers=["github"], providers_exclude=True), block
    )

    assert isinstance(response, ErrorResponse)
    assert "GitHub account" in response.message
    resolve.assert_not_awaited()
    execute.assert_not_awaited()
    db.delete_review_by_node_exec_id.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_a_block_that_cannot_be_run_directly_does_not_run_from_an_approval():
    response, resolve, execute, _ = await _resume_in_a_turn_with(
        None, _block("Run Workflow", BlockType.AGENT)
    )

    assert isinstance(response, ErrorResponse)
    assert "cannot be run directly" in response.message
    resolve.assert_not_awaited()
    execute.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_continuation_resolves_credentials_under_the_experts_grants():
    """The approved continuation re-resolves credentials, so it must be held
    to the expert's grants like the original run_block call."""
    tool = ContinueRunBlockTool()
    session = make_session(user_id=_TEST_USER_ID, expert_id="expert-a")
    review_id = "copilot-node-some-block:abc12345"
    review = _make_review_model(review_id, session_id=session.session_id)
    mock_db = MagicMock()
    mock_db.get_reviews_by_node_exec_ids = AsyncMock(return_value={review_id: review})
    block = MagicMock()
    block.name = "Some Block"
    block.disabled = False

    with (
        patch(
            "backend.copilot.tools.continue_run_block.review_db",
            return_value=mock_db,
        ),
        patch("backend.copilot.tools.continue_run_block.get_block", return_value=block),
        patch(
            "backend.copilot.tools.continue_run_block.resolve_block_credentials",
            new_callable=AsyncMock,
            return_value=({}, {"credentials": {}}),
        ) as resolve,
    ):
        await tool._execute(user_id=_TEST_USER_ID, session=session, review_id=review_id)

    resolve.assert_awaited_once()
    assert resolve.await_args_list[0].args[3] == "expert-a"
