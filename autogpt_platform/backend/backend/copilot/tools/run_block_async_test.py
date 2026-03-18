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
async def test_run_block_async_block_not_found(
    run_block_async_tool: RunBlockAsyncTool,
) -> None:
    with patch("backend.copilot.tools.run_block_async.get_block", return_value=None):
        result = await run_block_async_tool._execute(
            user_id="user1",
            session=_make_session(),
            block_id="nonexistent-id",
            block_name="Test",
            input_data={},
        )
    assert isinstance(result, ErrorResponse)
    assert "not found" in result.message


@pytest.mark.asyncio
async def test_run_block_async_returns_job_started(
    run_block_async_tool: RunBlockAsyncTool,
) -> None:
    mock_block = MagicMock()
    mock_block.disabled = False
    mock_block.block_type = MagicMock()
    mock_block.id = "block-uuid"
    mock_block.name = "Test Block"

    schema_mock = MagicMock()
    schema_mock.jsonschema.return_value = {
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    schema_mock.get_credentials_fields.return_value = {}
    schema_mock.get_credentials_fields_info.return_value = {}
    mock_block.input_schema = schema_mock
    mock_block.is_block_exec_need_review = AsyncMock(
        return_value=(False, {"text": "hello"})
    )

    jobs: dict = {}

    with (
        patch(
            "backend.copilot.tools.run_block_async.get_block", return_value=mock_block
        ),
        patch(
            "backend.copilot.tools.run_block_async.COPILOT_EXCLUDED_BLOCK_TYPES", set()
        ),
        patch(
            "backend.copilot.tools.run_block_async.COPILOT_EXCLUDED_BLOCK_IDS", set()
        ),
        patch(
            "backend.copilot.tools.run_block_async.resolve_block_credentials",
            AsyncMock(return_value=({}, set())),
        ),
        patch(
            "backend.copilot.tools.run_block_async.review_db",
            return_value=MagicMock(
                get_pending_reviews_for_execution=AsyncMock(return_value=[])
            ),
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
    # job should be cleaned up
    assert "job-123" not in jobs
