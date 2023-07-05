from unittest.mock import MagicMock

import pytest

from autogpt.agents import Agent
from autogpt.config import AIConfig
from autogpt.config.config import Config


@pytest.fixture
def agent(config: Config):
    ai_config = AIConfig(ai_name="Test AI")
    command_registry = MagicMock()
    memory = MagicMock()
    triggering_prompt = "Triggering prompt"
    workspace_directory = "workspace_directory"

    agent = Agent(
        ai_config=ai_config,
        command_registry=command_registry,
        memory=memory,
        triggering_prompt=triggering_prompt,
        workspace_directory=workspace_directory,
        config=config,
    )
    return agent


def test_agent_initialization(agent: Agent):
    assert agent.ai_config.ai_name == "Test AI"
    assert agent.history.messages == []
    assert agent.cycle_budget is None
    assert "You are Test AI" in agent.system_prompt


# More test methods can be added for specific agent interactions
# For example, mocking chat_with_ai and testing the agent's interaction loop
