"""Tests for run_block_async and get_block_result tools."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.get_block_result import GetBlockResultTool
from backend.copilot.tools.models import (
    BlockJobResultResponse,
    BlockJobStartedResponse,
    ErrorResponse,
    ResponseType,
)
from backend.copilot.tools.run_block_async import RunBlockAsyncTool


def _make_session(session_id: str = "test-session") -> MagicMock:
    session = MagicMock()
    session.session_id = session_id
    return session


@pytest.fixture()
def run_block_async_tool() -> RunBlockAsyncTool:
    return RunBlockAsyncTool()


@pytest.fixture()
def get_block_result_tool() -> GetBlockResultTool:
    return GetBlockResultTool()


@pytest.mark.asyncio
async def test_run_block_async_missing_block_id(
    run_block_async_tool: RunBlockAsyncTool,
) -> None:
    result = await run_block_async_tool._execute(
        user_id="user1",
        session=_make_session(),
        block_id="",
        block_name="Test",
        input_data={},
    )
    assert isinstance(result, ErrorResponse)
    assert result.type == ResponseType.ERROR


@pytest.mark.asyncio
async def test_run_block_async_no_auth(run_block_async_tool: RunBlockAsyncTool) -> None:
    result = await run_block_async_tool._execute(
        user_id=None,
        session=_make_session(),
        block_id="some-block-id",
        block_name="Test",
        input_data={},
    )
    assert isinstance(result, ErrorResponse)
    assert "Authentication" in result.message


@pytest.mark.asyncio
async def test_run_block_async_prepare_error_propagated(
    run_block_async_tool: RunBlockAsyncTool,
) -> None:
    """prepare_block_for_execution errors (e.g. block not found) are returned as-is."""
    with patch(
        "backend.copilot.tools.run_block_async.prepare_block_for_execution",
        AsyncMock(
            return_value=ErrorResponse(
                message="Block 'nonexistent-id' not found", session_id="test-session"
            )
        ),
    ):
        result = await run_block_async_tool._execute(
            user_id="user1",
            session=_make_session(),
            block_id="nonexistent-id",
            block_name="Test",
            input_data={"text": "hello"},
        )
    assert isinstance(result, ErrorResponse)
    assert "not found" in result.message


@pytest.mark.asyncio
async def test_run_block_async_missing_required_inputs(
    run_block_async_tool: RunBlockAsyncTool,
) -> None:
    """Returns ErrorResponse when required inputs are missing (no schema preview for async)."""
    mock_prep = MagicMock()
    mock_prep.block.name = "Test Block"
    mock_prep.required_non_credential_keys = {"text"}
    mock_prep.provided_input_keys = set()

    with patch(
        "backend.copilot.tools.run_block_async.prepare_block_for_execution",
        AsyncMock(return_value=mock_prep),
    ):
        result = await run_block_async_tool._execute(
            user_id="user1",
            session=_make_session(),
            block_id="block-uuid",
            block_name="Test Block",
            input_data={},
        )
    assert isinstance(result, ErrorResponse)
    assert "missing required inputs" in result.message.lower()


@pytest.mark.asyncio
async def test_run_block_async_returns_job_started(
    run_block_async_tool: RunBlockAsyncTool,
) -> None:
    mock_prep = MagicMock()
    mock_prep.block.name = "Test Block"
    mock_prep.block_id = "block-uuid"
    mock_prep.required_non_credential_keys = {"text"}
    mock_prep.provided_input_keys = {"text"}
    mock_prep.matched_credentials = {}

    jobs: dict = {}

    with (
        patch(
            "backend.copilot.tools.run_block_async.prepare_block_for_execution",
            AsyncMock(return_value=mock_prep),
        ),
        patch(
            "backend.copilot.tools.run_block_async.check_hitl_review",
            AsyncMock(return_value=("node-exec-id", {"text": "hello"})),
        ),
        patch(
            "backend.copilot.tools.run_block_async.execute_block",
            AsyncMock(
                return_value=MagicMock(
                    block_id="block-uuid",
                    block_name="Test Block",
                    outputs={},
                    success=True,
                )
            ),
        ),
        patch(
            "backend.copilot.sdk.tool_adapter.get_background_jobs",
            return_value=jobs,
        ),
    ):
        result = await run_block_async_tool._execute(
            user_id="user1",
            session=_make_session(),
            block_id="block-uuid",
            block_name="Test Block",
            input_data={"text": "hello"},
        )

    assert isinstance(result, BlockJobStartedResponse)
    assert result.type == ResponseType.BLOCK_JOB_STARTED
    assert result.job_id
    assert result.block_id == "block-uuid"
    # task must be stored so get_block_result can retrieve it
    assert len(jobs) == 1
    assert result.job_id in jobs


@pytest.mark.asyncio
async def test_get_block_result_missing_job(
    get_block_result_tool: GetBlockResultTool,
) -> None:
    with patch(
        "backend.copilot.sdk.tool_adapter.get_background_jobs",
        return_value={},
    ):
        result = await get_block_result_tool._execute(
            user_id="user1",
            session=_make_session(),
            job_id="nonexistent-job",
        )
    assert isinstance(result, ErrorResponse)
    assert "No background job" in result.message


@pytest.mark.asyncio
async def test_get_block_result_returns_output(
    get_block_result_tool: GetBlockResultTool,
) -> None:
    mock_output = MagicMock()
    mock_output.block_id = "block-uuid"
    mock_output.block_name = "Test Block"
    mock_output.outputs = {"result": ["value"]}
    mock_output.success = True

    async def _done() -> MagicMock:
        return mock_output

    task = asyncio.create_task(_done())
    await asyncio.sleep(0)  # let it run

    jobs = {"job-123": task}

    with patch(
        "backend.copilot.sdk.tool_adapter.get_background_jobs",
        return_value=jobs,
    ):
        result = await get_block_result_tool._execute(
            user_id="user1",
            session=_make_session(),
            job_id="job-123",
        )

    assert isinstance(result, BlockJobResultResponse)
    assert result.type == ResponseType.BLOCK_JOB_RESULT
    assert result.outputs == {"result": ["value"]}
    assert result.success is True
    # job should be cleaned up after retrieval
    assert "job-123" not in jobs
