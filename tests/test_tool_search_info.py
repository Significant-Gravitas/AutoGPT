import pytest
import os
from unittest.mock import AsyncMock, MagicMock
from AFAAS.lib.utils.json_schema import JSONSchema

# Assuming the not_implemented_tool function is defined in a module named `tools`
from AFAAS.core.tools.builtins.search_info import search_info, AbstractChatModelResponse
from AFAAS.core.tools.tool_decorator import tool
from AFAAS.interfaces.agent.main import BaseAgent
from tests.dataset.plan_familly_dinner import (
    Task,
    plan_familly_dinner,
    plan_step_0,
    task_ready_no_predecessors_or_subtasks,
    default_task
)



## This unit TEST TEST a LLM query
@pytest.mark.asyncio
async def test_search_info_none_command(default_task : Task,):
    mock_agent = MagicMock()
    mock_task = MagicMock()
    mock_assistant_response = MagicMock()
    mock_search_result = MagicMock(spec=AbstractChatModelResponse)
    mock_search_result.parsed_result = [None, None, {"response": "info"}]

    # Mock agent.execute_strategy to return a mock search result
    mock_agent.execute_strategy = AsyncMock(return_value=mock_search_result)

    result = await search_info(query="query", task=mock_task, agent=mock_agent)
    assert result == {"response": "info"}


@pytest.mark.asyncio
async def test_search_info_response_handling():
    # Setup mock objects
    mock_agent = default_task.agent
    mock_task = default_task
    mock_assistant_response = MagicMock()
    mock_search_result = MagicMock(spec=AbstractChatModelResponse)
    action =  'user_interaction'
    action_tool = mock_agent.tool_registry.get_tool("user_interaction")
    mock_parameters = {param.name: MagicMock() for param in action_tool.parameters}

    mock_search_result.parsed_result = [action, mock_parameters, mock_assistant_response]

    # Mock agent.execute_strategy to return the prepared search result
    mock_agent.execute_strategy = AsyncMock(return_value=mock_search_result)
    mock_agent.user_interaction = AsyncMock(return_value="User Interaction Result")

    # Call the search_info tool
    result = await search_info(query= "query", task=mock_task, agent=mock_agent)
    mock_agent.user_interaction.assert_called_once_with(task=mock_task, agent=mock_agent, additional="info")

    # Check the result
    assert result == "User Interaction Result"

@pytest.mark.asyncio
async def test_search_info_query_language_model_command(default_task : Task,):
    mock_agent = default_task.agent
    mock_task = default_task
    mock_assistant_response = MagicMock()
    mock_search_result = MagicMock(spec=AbstractChatModelResponse)
    action =  'query_language_model'
    action_tool = mock_agent.tool_registry.get_tool("query_language_model")
    mock_parameters = {param.name: MagicMock() for param in action_tool.parameters}

    mock_search_result.parsed_result = [action, mock_parameters, mock_assistant_response]

    mock_agent.execute_strategy = AsyncMock(return_value=mock_search_result)
    mock_agent.query_language_model = AsyncMock(return_value="LLM Result")

    result = await search_info(query="query", task=mock_task, agent=mock_agent)

    assert result == "LLM Result"
    mock_agent.query_language_model.assert_called_once()


@pytest.mark.asyncio
async def test_search_info_user_interaction_command(default_task : Task,):
    mock_agent = default_task.agent
    mock_task = default_task
    mock_assistant_response = MagicMock()
    mock_search_result = MagicMock(spec=AbstractChatModelResponse)
    action =  'user_interaction'
    action_tool = mock_agent.tool_registry.get_tool("user_interaction")
    mock_parameters = {param.name: MagicMock() for param in action_tool.parameters}

    mock_search_result.parsed_result = [action, mock_parameters, mock_assistant_response]

    mock_agent.execute_strategy = AsyncMock(return_value=mock_search_result)
    mock_agent.user_interaction = AsyncMock(return_value="Interaction Result")

    result = await search_info(query="query", task=mock_task, agent=mock_agent)

    assert result == "Interaction Result"



@pytest.mark.asyncio
async def test_search_info_web_search_command(default_task : Task,):
    mock_agent = default_task.agent
    mock_task = default_task
    mock_assistant_response = MagicMock()
    mock_search_result = MagicMock(spec=AbstractChatModelResponse)
    action =  'web_search'
    action_tool = mock_agent.tool_registry.get_tool("web_search")
    mock_parameters = {param.name: MagicMock() for param in action_tool.parameters}

    mock_search_result.parsed_result = [action, mock_parameters, mock_assistant_response]

    mock_agent.execute_strategy = AsyncMock(return_value=mock_search_result)

    with pytest.raises(NotImplementedError):
        await search_info(query="query", task=mock_task, agent=mock_agent)

@pytest.mark.asyncio
async def test_search_info_unrecognized_command(default_task : Task,):
    mock_agent = default_task.agent
    mock_task = default_task
    mock_assistant_response = MagicMock()
    mock_search_result = MagicMock(spec=AbstractChatModelResponse)
    action =  'unknown_command'
    action_tool = mock_agent.tool_registry.get_tool("unknown_command")
    mock_parameters = {param.name: MagicMock() for param in action_tool.parameters}

    mock_search_result.parsed_result = [action, mock_parameters, mock_assistant_response]

    mock_agent.execute_strategy = AsyncMock(return_value=mock_search_result)

    with pytest.raises(NotImplementedError):
        await search_info(query="query", task=mock_task, agent=mock_agent)


@pytest.mark.asyncio
async def test_search_info_strategy_exception():
    mock_agent = MagicMock()
    mock_task = MagicMock()
    mock_assistant_response = MagicMock()
    mock_agent.execute_strategy = AsyncMock(side_effect=Exception("Strategy error"))

    with pytest.raises(Exception) as exc_info:
        await search_info(query="query", task=mock_task, agent=mock_agent)

    assert "Strategy error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_search_info_invalid_args():
    mock_agent = MagicMock()
    mock_task = MagicMock()
    mock_assistant_response = MagicMock()

    with pytest.raises(TypeError):
        await search_info(query= 123, task=mock_task, agent=mock_agent)  # Invalid query type

@pytest.mark.asyncio
async def test_search_info_e2e(activate_integration_tests : bool , default_task : Task ):
    if not activate_integration_tests:
        pytest.skip("Skipping integration tests")

    default_task.task_context = "Some more information about the task"
    default_task.rag_history_txt = "This is what happened previously"
    default_task.rag_related_task_txt = "This are related things"

    # Define a query that will lead to a known command response
    test_query = "Mirror mirror on the wall, who is the fairest of them all?"
    agent = default_task.agent

    # Execute the search_info tool
    result = await search_info(query=test_query, task=default_task, agent=agent)

    # Assert that the result is a string
    assert isinstance(result, str)
