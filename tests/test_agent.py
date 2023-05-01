from unittest.mock import MagicMock

import pytest

from autogpt.agent import Agent
from autogpt.config import AIConfig
from autogpt.logs import Logger


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


def test_agent_initialization(agent):
    assert agent.ai_name == "Test AI"
    assert agent.memory == agent.memory
    assert agent.full_message_history == []
    assert agent.next_action_count == 0
    assert agent.command_registry == agent.command_registry
    assert agent.config == agent.config
    assert agent.system_prompt == "System prompt"
    assert agent.triggering_prompt == "Triggering prompt"


@pytest.fixture
def mock_input(mocker):
    mock = mocker.patch("builtins.input")
    mock.return_value = "y"
    return mock


def test_execute_command(agent, mocker, config, mock_input):
    mock_execute_command = mocker.patch("autogpt.agent.agent.execute_command")
    mock_execute_command.return_value = "command return value"
    mock_chat_with_ai = mocker.patch("autogpt.agent.agent.chat_with_ai")
    mock_chat_with_ai.return_value = """{
    "thoughts": {
        "text": "I will start by browsing the website www.steinhaug.no to gather information and save it to a file.",
        "reasoning": "The website is likely to have information about Steinhaug ut på eventyr, which is the topic of my report. Browsing the website is a good starting point for gathering information.",
        "plan": "- Browse www.steinhaug.no\\n- Save information to file\\n- Repeat for YouTube channel\\n- Write structured report",
        "criticism": "I need to ensure that I am not gathering irrelevant information from the website and that I am saving the information in a structured manner.",
        "speak": "I will start by browsing the website www.steinhaug.no to gather information and save it to a file."
    },
    "command": {
        "name": "browse_website",
        "args": {
            "url": "www.steinhaug.no",
            "question": "all information"
        }
    }
}"""

    mock_logger_info = mocker.patch.object(Logger, "info")
    mock_logger_typewriter_log = mocker.patch.object(Logger, "typewriter_log")
    mock_logger_warn = mocker.patch.object(Logger, "warn")

    config.continuous_limit = 1
    config.continuous_mode = True
    agent.start_interaction_loop()

    expected_execute_command = [
        mocker.call(
            agent.command_registry,
            "browse_website",
            {"question": "all information", "url": "www.steinhaug.no"},
            None,
        ),
    ]

    actual_execute_command_calls = mock_execute_command.call_args_list
    assert actual_execute_command_calls == expected_execute_command

    expected_logger_info = []
    actual_logger_info_calls = mock_logger_info.call_args_list
    assert actual_logger_info_calls == expected_logger_info

    expected_typewriter_log = [
        mocker.call(
            "TEST AI THOUGHTS:",
            "\x1b[33m",
            "I will start by browsing the website www.steinhaug.no to gather information and save it to a file.",
        ),
        mocker.call(
            "REASONING:",
            "\x1b[33m",
            "The website is likely to have information about Steinhaug ut på eventyr, which is the topic of my report. Browsing the website is a good starting point for gathering information.",
        ),
        mocker.call("PLAN:", "\x1b[33m", ""),
        mocker.call("- ", "\x1b[32m", "Browse www.steinhaug.no"),
        mocker.call("- ", "\x1b[32m", "Save information to file"),
        mocker.call("- ", "\x1b[32m", "Repeat for YouTube channel"),
        mocker.call("- ", "\x1b[32m", "Write structured report"),
        mocker.call(
            "CRITICISM:",
            "\x1b[33m",
            "I need to ensure that I am not gathering irrelevant information from the website and that I am saving the information in a structured manner.",
        ),
        mocker.call(
            "NEXT ACTION: ",
            "\x1b[36m",
            "COMMAND = \x1b[36mbrowse_website\x1b[0m  ARGUMENTS = \x1b[36m{'url': 'www.steinhaug.no', 'question': 'all information'}\x1b[0m",
        ),
        mocker.call(
            "SYSTEM: ",
            "\x1b[33m",
            "Command browse_website returned: command return value",
        ),
        mocker.call("Continuous Limit Reached: ", "\x1b[33m", "1"),
    ]

    actual_logger_typewriter_log_calls = mock_logger_typewriter_log.call_args_list
    assert actual_logger_typewriter_log_calls == expected_typewriter_log

    expected_logger_warn = []
    actual_logger_warn_calls = mock_logger_warn.call_args_list
    assert actual_logger_warn_calls == expected_logger_warn
