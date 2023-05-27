from unittest.mock import MagicMock

import pytest

from autogpt.agent import Agent
from autogpt.agent.agent import is_command_result_an_error
from autogpt.config import AIConfig


@pytest.fixture
def agent():
    ai_name = "Test AI"
    memory = MagicMock()
    next_action_count = 0
    command_registry = MagicMock()
    config = AIConfig()
    system_prompt = "System prompt"
    triggering_prompt = "Triggering prompt"
    workspace_directory = "workspace_directory"

    agent = Agent(
        ai_name,
        memory,
        next_action_count,
        command_registry,
        config,
        system_prompt,
        triggering_prompt,
        workspace_directory,
    )
    return agent


def test_agent_initialization(agent: Agent):
    assert agent.ai_name == "Test AI"
    assert agent.memory == agent.memory
    assert agent.history.messages == []
    assert agent.next_action_count == 0
    assert agent.command_registry == agent.command_registry
    assert agent.config == agent.config
    assert agent.system_prompt == "System prompt"
    assert agent.triggering_prompt == "Triggering prompt"


def test_is_command_result_an_error():
    input_1 = "error"
    result_1 = is_command_result_an_error(input_1)
    assert result_1 is True

    input_2 = "unknown command"
    result_2 = is_command_result_an_error(input_2)
    assert result_2 is True

    input_3 = "traceback"
    result_3 = is_command_result_an_error(input_3)
    assert result_3 is True

    input_4 = "test"
    result_4 = is_command_result_an_error(input_4)
    assert result_4 is False


# More test methods can be added for specific agent interactions
# For example, mocking chat_with_ai and testing the agent's interaction loop
