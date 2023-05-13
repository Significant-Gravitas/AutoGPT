"""Unit tests for the commands module"""
from unittest.mock import MagicMock, patch

import pytest

from autogpt.app import list_agents, start_agent
from tests.utils import requires_api_key


@pytest.mark.vcr
@pytest.mark.integration_test
@requires_api_key("OPENAI_API_KEY")
def test_make_agent() -> None:
    """Test that an agent can be created"""
    # Use the mock agent manager to avoid creating a real agent
    with patch("openai.ChatCompletion.create") as mock:
        response = MagicMock()
        response.choices[0].messages[0].content = "Test message"
        response.usage.prompt_tokens = 1
        response.usage.completion_tokens = 1
        mock.return_value = response
        start_agent("Test Agent", "chat", "Hello, how are you?", "gpt-3.5-turbo")
        agents = list_agents()
        assert "List of agents:\n0: chat" == agents
        start_agent("Test Agent 2", "write", "Hello, how are you?", "gpt-3.5-turbo")
        agents = list_agents()
        assert "List of agents:\n0: chat\n1: write" == agents
