from unittest.mock import AsyncMock, MagicMock

import pytest

from AFAAS.core.tools.builtins.query_language_model import query_language_model
from AFAAS.interfaces.prompts.strategy import AbstractChatModelResponse
from tests.dataset.plan_familly_dinner import (
    Task,
    _plan_familly_dinner,
    default_task,
    plan_familly_dinner_with_tasks_saved_in_db,
    plan_step_0,
    task_ready_no_predecessors_or_subtasks,
)


@pytest.mark.asyncio
async def test_query_language_model_returns_string():
    # Mock the Task and BaseAgent objects
    mock_task = MagicMock()
    mock_agent = MagicMock()

    # Mock the execute_strategy method of the agent
    # It should return a string, as the function is expected to return a string
    mock_execute_strategy = AsyncMock()
    mock_execute_strategy.return_value.parsed_result = "Test Plan String"
    mock_agent.execute_strategy = mock_execute_strategy

    # Call the function with the mocked objects
    result = await query_language_model(
        query="How to plan a familly dinner ?",
        format="Aswer in 3 paragraphs",
        persona="A drunk Paraguayan sailor",
        task=mock_task,
        agent=mock_agent,
    )

    # Assert that the result is a string
    assert isinstance(result, str)
    assert result == "Test Plan String"


@pytest.mark.asyncio
async def test_query_language_model_integration(
    activate_integration_tests, default_task: Task
):
    if not activate_integration_tests:
        pytest.skip("Integration tests are not activated")

    # Call the function with the real or semi-real objects
    result = await query_language_model(
        task=default_task,
        agent=default_task.agent,
        query="How to plan a familly dinner ?",
        format="Aswer in 3 paragraphs",
        persona="A drunk Paraguayan sailor",
    )

    assert isinstance(result, str)
