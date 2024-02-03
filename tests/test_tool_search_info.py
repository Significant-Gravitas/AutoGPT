import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Assuming the not_implemented_tool function is defined in a module named `tools`
from AFAAS.core.tools.builtins.search_info import AbstractChatModelResponse, search_info
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
async def test_search_info_query_language_model_command(default_task: Task):
    pytest.skip()
    mock_agent = default_task.agent
    mock_task = default_task
    mock_search_result = MagicMock(spec=AbstractChatModelResponse)

    action = "query_language_model"
    action_tool = mock_agent.tool_registry.get_tool(action)
    mock_parameters = {param.name: MagicMock() for param in action_tool.parameters}
    mock_search_result.parsed_result = [action, mock_parameters, {}]

    expected_result = "LLM Result"
    mock_agent.execute_strategy = AsyncMock(
        side_effect=[mock_search_result, expected_result]
    )

    result = await search_info(
        query="query", reasoning="reasoning", task=mock_task, agent=mock_agent
    )
    assert result == expected_result


@pytest.mark.asyncio
async def test_search_info_user_interaction_command(
    default_task: Task,
):
    pytest.skip()
    mock_agent = default_task.agent
    mock_task = default_task
    mock_assistant_response = MagicMock()
    mock_search_result = MagicMock(spec=AbstractChatModelResponse)
    action = "user_interaction"
    action_tool = mock_agent.tool_registry.get_tool("user_interaction")
    mock_parameters = {param.name: MagicMock() for param in action_tool.parameters}

    mock_search_result.parsed_result = [
        action,
        mock_parameters,
        mock_assistant_response,
    ]

    expected_result = "Interaction Result"
    mock_agent.execute_strategy = AsyncMock(return_value=mock_search_result)
    with patch(
        "AFAAS.core.tools.builtins.search_info.user_interaction",
        new=AsyncMock(return_value=expected_result),
    ) as mock_user_interaction:
        # Call the search_info function
        result = await search_info(
            query="query", reasoning="reasoning", task=mock_task, agent=mock_agent
        )

        # Assert that the result matches the expected result from query_language_model
        assert result == expected_result
        mock_user_interaction.assert_called_once_with(
            task=mock_task, agent=mock_agent, **mock_parameters
        )


@pytest.mark.asyncio
async def test_search_info_web_search_command(
    default_task: Task,
):
    pytest.skip()
    mock_agent = default_task.agent
    mock_task = default_task
    mock_assistant_response = MagicMock()
    mock_search_result = MagicMock(spec=AbstractChatModelResponse)
    action = "web_search"
    action_tool = mock_agent.tool_registry.get_tool("web_search")
    mock_parameters = {param.name: MagicMock() for param in action_tool.parameters}

    mock_search_result.parsed_result = [
        action,
        mock_parameters,
        mock_assistant_response,
    ]

    mock_agent.execute_strategy = AsyncMock(return_value=mock_search_result)

    with pytest.raises(NotImplementedError):
        await search_info(
            query="query", reasoning="reasoning", task=mock_task, agent=mock_agent
        )


@pytest.mark.asyncio
async def test_search_info_unrecognized_command(
    default_task: Task,
):
    pytest.skip()
    mock_agent = default_task.agent
    mock_task = default_task
    mock_assistant_response = MagicMock()
    mock_search_result = MagicMock(spec=AbstractChatModelResponse)
    action = "unknown_command"

    action_tool = MagicMock()
    mock_parameters = MagicMock()

    mock_search_result.parsed_result = [{ "command_name" : action,
        "command_args": mock_parameters,
        "mock_assistant_response" : mock_assistant_response
        }
    ]

    mock_agent.execute_strategy = AsyncMock(return_value=mock_search_result)

    with pytest.raises(NotImplementedError):
        await search_info(
            query="query", reasoning="reasoning", task=mock_task, agent=mock_agent
        )


@pytest.mark.asyncio
async def test_search_info_strategy_exception():
    mock_agent = MagicMock()
    mock_task = MagicMock()
    mock_assistant_response = MagicMock()
    mock_agent.execute_strategy = AsyncMock(side_effect=Exception("Strategy error"))

    with pytest.raises(Exception) as exc_info:
        await search_info(
            query="query", reasoning="reasoning", task=mock_task, agent=mock_agent
        )

    assert "Strategy error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_search_info_invalid_args():
    mock_agent = MagicMock()
    mock_task = MagicMock()
    mock_assistant_response = MagicMock()

    with pytest.raises(TypeError):
        await search_info(
            query=123, reasoning="reasoning", task=mock_task, agent=mock_agent
        )  # Invalid query type


@pytest.mark.asyncio
async def test_search_info_e2e(activate_integration_tests: bool, default_task: Task):
    if not activate_integration_tests and False:
        pytest.skip("Skipping integration tests")

    default_task.task_context = "Some more information about the task"
    default_task.rag_history_txt = "This is what happened previously"
    default_task.rag_related_task_txt = "This are related things"

    # Define a query that will lead to a known command response
    test_query = "Mirror mirror on the wall, who is the fairest of them all?"
    reasoning = "I am trying to find out who is the fairest of them all"
    agent = default_task.agent

    # Execute the search_info tool
    result = await search_info(
        query=test_query, task=default_task, agent=agent, reasoning=reasoning
    )

    # Assert that the result is a string
    assert isinstance(result, str)
