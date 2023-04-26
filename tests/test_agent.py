from unittest.mock import MagicMock

import pytest

from autogpt.agent import Agent
from autogpt.chat import chat_with_ai
from autogpt.config import Config
from autogpt.speech import say_text
from autogpt.utils import clean_input


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


class CustomPlugin:
    def __init__(self, should_handle):
        self.should_handle = should_handle

    def can_handle_post_command(self):
        return self.should_handle

    def post_command(self, command_name, result):
        return f"Modified: {result}"


def test_plugin_post_command_handling(agent):
    # Test with a plugin that handles post_command
    plugin = CustomPlugin(should_handle=True)
    agent.config.plugins.append(plugin)

    command_name = "test_command"
    result = "Test result"

    modified_result = agent._handle_plugin_post_command(command_name, result)
    assert modified_result == f"Modified: {result}"

    # Test with a plugin that doesn't handle post_command
    plugin.should_handle = False
    unmodified_result = agent._handle_plugin_post_command(command_name, result)
    assert unmodified_result == result
