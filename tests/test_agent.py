from unittest import mock
from unittest.mock import MagicMock

import pytest

from autogpt.agent import Agent
from autogpt.config import AIConfig
from autogpt.config.config import Config
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


def test_agent_initialization(agent: Agent):
    assert agent.ai_name == "Test AI"
    assert agent.memory == agent.memory
    assert agent.history.messages == []
    assert agent.next_action_count == 0
    assert agent.command_registry == agent.command_registry
    assert agent.config == agent.config
    assert agent.system_prompt == "System prompt"
    assert agent.triggering_prompt == "Triggering prompt"


def test_chat_with_ai_model_is_none(config: Config, agent: Agent):
    with mock.patch(
        "autogpt.llm.chat.create_chat_completion"
    ) as mock_create_chat_completion:
        mock_create_chat_completion.return_value = "test"
        with mock.patch(
            "autogpt.llm.chat.count_message_tokens"
        ) as mock_count_message_tokens:
            mock_count_message_tokens.return_value = 3000
            chat_with_ai(
                config=config,
                agent=agent,
                user_input="test",
                token_limit=3000,
                system_prompt="System prompt",
            )

            mock_count_message_tokens.assert_called()
            args_list = mock_count_message_tokens.call_args
            print(args_list)
            assert any(
                "gpt-4" in arg_tuple[1] for arg_tuple in args_list
            ), "gpt-4 not found in args_list"


# More test methods can be added for specific agent interactions
# For example, mocking chat_with_ai and testing the agent's interaction loop
