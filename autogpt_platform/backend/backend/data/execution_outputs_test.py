"""Test that get_execution_outputs_by_node_exec_id returns CompletedBlockOutput.

CompletedBlockOutput is dict[str, list[Any]] — values must be lists.
The RPC service layer validates return types via TypeAdapter, so if
the function returns plain values instead of lists, it causes:

    1 validation error for dict[str,list[any]] response
    Input should be a valid list [type=list_type, input_value='', input_type=str]

This breaks SmartDecisionMakerBlock agent mode tool execution.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import TypeAdapter

from backend.data.block import CompletedBlockOutput


@pytest.mark.asyncio
async def test_outputs_are_lists():
    """Each value in the returned dict must be a list, matching CompletedBlockOutput."""
    from backend.data.execution import get_execution_outputs_by_node_exec_id

    mock_output = MagicMock()
    mock_output.name = "response"
    mock_output.data = "some text output"

    with patch(
        "backend.data.execution.AgentNodeExecutionInputOutput.prisma"
    ) as mock_prisma:
        mock_prisma.return_value.find_many = AsyncMock(return_value=[mock_output])
        result = await get_execution_outputs_by_node_exec_id("test-exec-id")

    # The result must conform to CompletedBlockOutput = dict[str, list[Any]]
    assert "response" in result
    assert isinstance(
        result["response"], list
    ), f"Expected list, got {type(result['response']).__name__}: {result['response']!r}"

    # Must also pass TypeAdapter validation (this is what the RPC layer does)
    adapter = TypeAdapter(CompletedBlockOutput)
    validated = adapter.validate_python(result)  # This is the line that fails in prod
    assert validated == {"response": ["some text output"]}


@pytest.mark.asyncio
async def test_multiple_outputs_same_name_are_collected():
    """Multiple outputs with the same name should all appear in the list."""
    from backend.data.execution import get_execution_outputs_by_node_exec_id

    mock_out1 = MagicMock()
    mock_out1.name = "result"
    mock_out1.data = "first"

    mock_out2 = MagicMock()
    mock_out2.name = "result"
    mock_out2.data = "second"

    with patch(
        "backend.data.execution.AgentNodeExecutionInputOutput.prisma"
    ) as mock_prisma:
        mock_prisma.return_value.find_many = AsyncMock(
            return_value=[mock_out1, mock_out2]
        )
        result = await get_execution_outputs_by_node_exec_id("test-exec-id")

    assert isinstance(result["result"], list)
    assert len(result["result"]) == 2


@pytest.mark.asyncio
async def test_empty_outputs_returns_empty_dict():
    """No outputs → empty dict."""
    from backend.data.execution import get_execution_outputs_by_node_exec_id

    with patch(
        "backend.data.execution.AgentNodeExecutionInputOutput.prisma"
    ) as mock_prisma:
        mock_prisma.return_value.find_many = AsyncMock(return_value=[])
        result = await get_execution_outputs_by_node_exec_id("test-exec-id")

    assert result == {}


@pytest.mark.asyncio
async def test_none_data_skipped():
    """Outputs with data=None should be skipped."""
    from backend.data.execution import get_execution_outputs_by_node_exec_id

    mock_output = MagicMock()
    mock_output.name = "response"
    mock_output.data = None

    with patch(
        "backend.data.execution.AgentNodeExecutionInputOutput.prisma"
    ) as mock_prisma:
        mock_prisma.return_value.find_many = AsyncMock(return_value=[mock_output])
        result = await get_execution_outputs_by_node_exec_id("test-exec-id")

    assert result == {}
