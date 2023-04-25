from unittest.mock import MagicMock

import pytest

from autogpt.agent import Agent
from autogpt.chat import chat_with_ai
from autogpt.config import Config
from autogpt.speech import say_text
from autogpt.utils import clean_input
from autogpt.ai_guidelines import AIGuidelines


@pytest.fixture
def agent():
    ai_name = "Test AI"
    memory = MagicMock()
    full_message_history = []
    next_action_count = 0
    command_registry = MagicMock()
    config = Config()
    system_prompt = "System prompt"
    triggering_prompt = "Triggering prompt"
    workspace_directory = "workspace_directory"
    guidelines_mgr = AIGuidelines("ai_guidelines.yaml", bsilent=True)


    agent = Agent(
        ai_name,
        memory,
        full_message_history,
        next_action_count,
        command_registry,
        config,
        system_prompt,
        triggering_prompt,
        workspace_directory,
        guidelines_mgr,
    )
    return agent


def test_agent_initialization(agent):
    assert agent.ai_name == "Test AI"
    assert agent.memory == agent.memory
    assert agent.full_message_history == []
    assert agent.next_action_count == 0
    assert agent.command_registry == agent.command_registry
    assert agent.config == agent.config
    assert agent.system_prompt == "System prompt"
    assert agent.triggering_prompt == "Triggering prompt"
    assert agent.ai_guidelines == agent.ai_guidelines


# More test methods can be added for specific agent interactions
# For example, mocking chat_with_ai and testing the agent's interaction loop
