from unittest.mock import MagicMock

from pytest_mock import MockerFixture

from autogpt.agent.agent import Agent
from autogpt.app import list_agents, start_agent


def test_make_agent(agent: Agent, mocker: MockerFixture) -> None:
    """Test that an agent can be created"""
    mock = mocker.patch("openai.ChatCompletion.create")

    response = MagicMock()
    del response.error
    response.choices[0].messages[0].content = "Test message"
    response.usage.prompt_tokens = 1
    response.usage.completion_tokens = 1
    mock.return_value = response
    start_agent("Test Agent", "chat", "Hello, how are you?", agent, "gpt-3.5-turbo")
    agents = list_agents(agent)
    assert "List of agents:\n0: chat" == agents
    start_agent("Test Agent 2", "write", "Hello, how are you?", agent, "gpt-3.5-turbo")
    agents = list_agents(agent.config)
    assert "List of agents:\n0: chat\n1: write" == agents
