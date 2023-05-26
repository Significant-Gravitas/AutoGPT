from unittest.mock import MagicMock

import pytest

from autogpt.agent import Agent
from autogpt.config.ai_config import AIConfig
from autogpt.llm.base import Message
from autogpt.llm.chat import chat_with_ai


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


def test_chat_with_ai_basic_response(mocker, agent):
    prompt = "Welcome to the autogpt!"
    user_input = "Hello, how are you?"
    token_limit = 4000
    mocker.patch(
        "autogpt.llm.chat.create_chat_completion",
        return_value="I'm doing well, thank you for asking.",
    )

    response = chat_with_ai(agent, prompt, user_input, token_limit)

    assert response == "I'm doing well, thank you for asking."
    assert agent.history.messages == [
        Message("user", "Hello, how are you?", None),
        Message("assistant", "I'm doing well, thank you for asking.", "ai_response"),
    ]
