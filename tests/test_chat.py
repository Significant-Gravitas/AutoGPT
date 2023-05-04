from unittest.mock import MagicMock

import pytest

from autogpt.agent import Agent
from autogpt.config.ai_config import AIConfig
from autogpt.llm.chat import chat_with_ai


@pytest.fixture
def agent():
    ai_name = "Test AI"
    memory = MagicMock()
    full_message_history = []
    next_action_count = 0
    command_registry = MagicMock()
    config = AIConfig()
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


def test_chat_with_ai_basic_response(mocker, agent):
    prompt = "Welcome to the autogpt!"
    user_input = "Hello, how are you?"
    full_message_history = []
    permanent_memory = agent.memory
    token_limit = 4000
    mocker.patch(
        "autogpt.llm.chat.create_chat_completion",
        return_value="I'm doing well, thank you for asking.",
    )

    response = chat_with_ai(
        agent, prompt, user_input, full_message_history, permanent_memory, token_limit
    )

    assert response == "I'm doing well, thank you for asking."
    assert full_message_history == [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking."},
    ]


def test_chat_with_ai_summary_memory(mocker, agent):
    prompt = "Welcome to the autogpt!"
    user_input = "Hello, how are you?"
    full_message_history = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking."},
    ]
    permanent_memory = agent.memory
    token_limit = 4000
    mocker.patch(
        "autogpt.llm.chat.create_chat_completion",
        return_value="I'm doing well, thank you for asking.",
    )
    mocker.patch(
        "autogpt.memory_management.summary_memory.create_chat_completion",
        return_value="I was created and nothing new has happened.",
    )

    response = chat_with_ai(
        agent, prompt, user_input, full_message_history, permanent_memory, token_limit
    )

    assert response == "I'm doing well, thank you for asking."
    assert agent.summary_memory == {
        "role": "system",
        "content": "This reminds you of these events from your past: \nI was created and nothing new has happened.",
    }
