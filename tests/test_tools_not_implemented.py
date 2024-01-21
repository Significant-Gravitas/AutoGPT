import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Assuming the not_implemented_tool function is defined in a module named `tools`
from AFAAS.core.tools.builtins.not_implemented_tool import not_implemented_tool
from AFAAS.core.tools.tool_decorator import SAFE_MODE, tool
from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.lib.utils.json_schema import JSONSchema
from tests.dataset.plan_familly_dinner import (
    Task,
    _plan_familly_dinner,
    default_task,
    plan_familly_dinner_with_tasks_saved_in_db,
    plan_step_0,
    task_ready_no_predecessors_or_subtasks,
)


@pytest.mark.asyncio
async def test_not_implemented_tool_basic(default_task: Task):
    # Mock Task and BaseAgent
    mock_task = default_task
    mock_agent = default_task.agent

    # Assert that the result is what user_interaction returns
    expected_result = "Interaction Result"
    with patch(
        "AFAAS.core.tools.builtins.user_interaction.user_interaction",
        new=AsyncMock(return_value=expected_result),
    ) as mock_user_interaction:
        result = await not_implemented_tool(
            task=mock_task, agent=mock_agent, query="Test Query"
        )
        assert result == expected_result


# Use a fixture to determine whether to run integration tests
@pytest.fixture(scope="session")
def activate_integration_tests():
    return os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"


@pytest.mark.asyncio
async def test_async_tool_not_implemented(default_task: Task):
    @tool(name="async_test_tool", description="Async Test Tool")
    async def async_test_tool(agent: BaseAgent, task: Task) -> str:
        raise NotImplementedError

    mock_task = default_task
    mock_agent = default_task.agent
    # Assert that the result is what user_interaction returns
    expected_result = "Interaction Result"
    with patch(
        "AFAAS.core.tools.builtins.user_interaction.user_interaction",
        new=AsyncMock(return_value=expected_result),
    ) as mock_user_interaction:
        result = await async_test_tool(task=mock_task, agent=mock_agent)
        assert result == expected_result


def test_sync_tool_not_implemented(default_task: Task):
    @tool(name="sync_test_tool", description="Sync Test Tool")
    def sync_test_tool(agent: BaseAgent, task: Task) -> str:
        raise NotImplementedError

    mock_task = default_task
    mock_agent = default_task.agent

    # Assert that the result is what user_interaction returns
    expected_result = "Interaction Result"
    with patch(
        "AFAAS.core.tools.builtins.user_interaction.user_interaction",
        new=AsyncMock(return_value=expected_result),
    ) as mock_user_interaction:
        result = sync_test_tool(task=mock_task, agent=mock_agent)
        assert result == expected_result


@pytest.mark.asyncio
async def test_async_tool_with_args_not_implemented(default_task: Task):
    @tool(
        name="async_test_tool_args",
        description="Async Test Tool with Args",
        parameters={"query": JSONSchema(type=JSONSchema.Type.STRING, required=True)},
    )
    async def async_test_tool_args(query: str, agent: BaseAgent, task: Task) -> str:
        raise NotImplementedError

    mock_task = default_task
    mock_agent = default_task.agent

    # Assert that the result is what user_interaction returns
    expected_result = "Interaction Result"
    with patch(
        "AFAAS.core.tools.builtins.user_interaction.user_interaction",
        new=AsyncMock(return_value=expected_result),
    ) as mock_user_interaction:
        result = await async_test_tool_args(
            query="Test Query", task=mock_task, agent=mock_agent
        )
        assert result == expected_result


@pytest.mark.asyncio
async def test_async_tool_no_kwargs_not_implemented(default_task: Task):
    @tool(name="async_test_tool_no_kwargs", description="Async Test Tool No KWArgs")
    async def async_test_tool_no_kwargs(agent: BaseAgent, task: Task) -> str:
        raise NotImplementedError

    mock_task = default_task
    mock_agent = default_task.agent

    # Assert that the result is what user_interaction returns
    expected_result = "Interaction Result"
    with patch(
        "AFAAS.core.tools.builtins.user_interaction.user_interaction",
        new=AsyncMock(return_value=expected_result),
    ) as mock_user_interaction:
        result = await async_test_tool_no_kwargs(task=mock_task, agent=mock_agent)
        assert result == expected_result


@pytest.mark.asyncio
async def test_not_implemented_tool_integration(
    activate_integration_tests, task_ready_no_predecessors_or_subtasks
):
    if not activate_integration_tests:
        pytest.skip("Integration tests are not activated")

    # Here, you would set up real or semi-real Task and BaseAgent
    real_task = task_ready_no_predecessors_or_subtasks
    real_agent = task_ready_no_predecessors_or_subtasks.agent

    # Call the function with real or semi-real objects
    result = await not_implemented_tool(
        task=real_task, agent=real_agent, query="Integration Test Query"
    )

    # FIXME: Maxe assertions
    pytest.skip("Not Implemented Tool Integration Test Assertions Not Implemented")
