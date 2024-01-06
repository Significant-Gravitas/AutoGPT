import pytest
from unittest.mock import AsyncMock, MagicMock
from AFAAS.core.tools.builtins.query_language_model import query_language_model

@pytest.mark.asyncio
async def test_query_language_model_returns_string():
    # Mock the Task and BaseAgent objects
    mock_task = MagicMock()
    mock_agent = MagicMock()

    # Mock the execute_strategy method of the agent
    # It should return a string, as the function is expected to return a string
    mock_agent.execute_strategy = AsyncMock(return_value="Test Plan String")

    # Call the function with the mocked objects
    result = await query_language_model(
        query="How to plan a familly dinner ?",
        format="Aswer in 3 paragraphs",
        persona="A drunk Paraguayan sailor",
        task = mock_task, 
        agent = mock_agent
        )

    # Assert that the result is a string
    assert isinstance(result, str)
    assert result == "Test Plan String"


@pytest.mark.asyncio
async def test_query_language_model_integration(activate_integration_tests, mock_agent, mock_task):
    if not activate_integration_tests:
        pytest.skip("Integration tests are not activated")

    # Here, mock_agent.execute_strategy is not mocked
    # Add the necessary setup for mock_agent and mock_task for a real scenario

    # Call the function with the real or semi-real objects
    result = await query_language_model(mock_task, mock_agent)

