
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from AFAAS.core.tools.builtins.afaas_refine_user_context import afaas_refine_user_context, RefineUserContextFunctionNames
from AFAAS.lib.task.task import Task
from AFAAS.interfaces.agent.main import BaseAgent

@pytest.mark.asyncio
@patch('AFAAS.core.tools.builtins.user_interaction.user_interaction')
async def test_afaas_refine_user_context_success(mock_user_interaction, default_task : Task):
    # Setup mocks
    mock_task = default_task
    mock_agent = default_task.agent
    mock_agent.execute_strategy = AsyncMock()  # Mock strategy execution response

    # Configure the mock_user_interaction to return a user response
    mock_user_interaction.return_value = AsyncMock(return_value="User response")

    # Expected refined goals to be set in task memory
    expected_goal_sentence = "I want to learn advanced cooking"
    questions = ["What do you want to learn?", "What is your goal?"]
    expected_goals = ["Learn cooking basics", "Explore advanced recipes"]

    # Configure mock responses to simulate interaction flow
    mock_agent.execute_strategy.side_effect = [
        # Simulate refine requirements response
        MagicMock(parsed_result={"name": RefineUserContextFunctionNames.REFINE_REQUIREMENTS, "reformulated_goal": expected_goal_sentence,
        "questions": questions }),
        # Simulate validate requirements response
        MagicMock(parsed_result={"name": RefineUserContextFunctionNames.VALIDATE_REQUIREMENTS, "goal_list": expected_goals})
    ]

    # Call the function
    await afaas_refine_user_context(mock_task, mock_agent, user_objectives="I want to learn how to cook")

    # Verify task memory update
    assert mock_task.memory["agent_goal_sentence"] == expected_goal_sentence
    assert mock_task.memory["agent_goals"] == expected_goals

@pytest.mark.asyncio
async def test_afaas_refine_user_context_invalid_input():
    # Setup mocks
    mock_task = MagicMock(spec=Task)
    mock_agent = MagicMock(spec=BaseAgent)

    # Attempt to call the function with invalid inputs
    with pytest.raises(Exception) as exc_info:
        await afaas_refine_user_context(mock_task, mock_agent, user_objectives="")



@pytest.mark.asyncio
@patch('AFAAS.core.tools.builtins.user_interaction.user_interaction')
async def test_afaas_refine_user_context_interruption_and_confirmation(mock_user_interaction, default_task : Task):
    # Setup mocks
    mock_task = default_task
    mock_agent = default_task.agent
    mock_agent.execute_strategy = AsyncMock()  # Mock strategy execution response

    # Configure the mock_user_interaction to return a user response
    mock_user_interaction.return_value = AsyncMock(return_value="User response")

    # Expected refined goals to be set in task memory
    expected_goal_sentence = "I want to learn advanced cooking"
    questions = ["What do you want to learn?", "What is your goal?"]
    expected_goal_sentence_v2 = "I want to learn advanced cooking and explore new recipes"
    questions_v2 = ["Any cuisine preference?", "Do you have any dietary restrictions?"]
    expected_goals = ["Learn cooking basics", "Explore advanced recipes"]

    # Mock user interaction to simulate interruption and confirmation
    mock_agent.execute_strategy.side_effect = [
        # Simulate refine requirements response
        MagicMock(parsed_result={"name": RefineUserContextFunctionNames.REFINE_REQUIREMENTS, "reformulated_goal": expected_goal_sentence,
        "questions": questions }),
        # Simulate request for second confirmation
        MagicMock(parsed_result={"name": RefineUserContextFunctionNames.REQUEST_SECOND_CONFIRMATION, 
                                "questions" : questions_v2,
                                "reformulated_goal": expected_goal_sentence_v2}),
        # User confirms to proceed
        MagicMock(parsed_result={"name": RefineUserContextFunctionNames.VALIDATE_REQUIREMENTS, "goal_list": expected_goals})
    ]

    # Call the function with a user objective
    await afaas_refine_user_context(mock_task, mock_agent, user_objectives="Some initial goal")

    # Assert the process is completed successfully
    assert ["Learn cooking basics", "Explore advanced recipes"] == mock_task.memory["agent_goals"]
