"""Unit tests for the commands module"""
from unittest.mock import MagicMock, patch

import pytest

import autogpt.agent.agent_manager as agent_manager
from autogpt.app import execute_command, list_agents, start_agent


@pytest.mark.integration_test
def test_make_agent() -> None:
    """Test the make_agent command"""
    with patch("openai.ChatCompletion.create") as mock:
        obj = MagicMock()
        obj.response.choices[0].messages[0].content = "Test message"
        mock.return_value = obj
        start_agent("Test Agent", "chat", "Hello, how are you?", "gpt2")
        agents = list_agents()
        assert "List of agents:\n0: chat" == agents
        start_agent("Test Agent 2", "write", "Hello, how are you?", "gpt2")
        agents = list_agents()
        assert "List of agents:\n0: chat\n1: write" == agents
