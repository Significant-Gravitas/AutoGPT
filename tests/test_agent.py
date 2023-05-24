import functools
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


def single_agent_loop_setup(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        config = kwargs.get("config")
        original_continuous_limit = config.continuous_limit
        original_speak_mode = config.speak_mode
        original_continuous_mode = config.continuous_mode
        config.continuous_limit = 1
        config.continuous_mode = True
        config.speak_mode = True
        result = func(*args, **kwargs)
        config.continuous_limit = original_continuous_limit
        config.continuous_mode = original_continuous_mode
        config.speak_mode = original_speak_mode
        return result

    return wrapper


def setup_with_chat(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        config = kwargs.get("config")
        original_chat_messages_enabled = config.chat_messages_enabled
        config.chat_messages_enabled = True
        result = func(*args, **kwargs)
        config.chat_messages_enabled = original_chat_messages_enabled
        return result

    return wrapper


@pytest.fixture
def mock_input(mocker):
    mock = mocker.patch("builtins.input")
    mock.return_value = "y"
    return mock


@pytest.fixture
def mock_chat_with_ai(mocker):
    mock = mocker.patch("autogpt.agent.agent.chat_with_ai")
    mock.return_value = """{
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
    return mock


@pytest.fixture
def mock_execute_command(mocker):
    mock = mocker.patch("autogpt.agent.agent.execute_command")
    mock.return_value = "command return value"
    return mock


@pytest.fixture
def mock_logger_info(mocker):
    return mocker.patch.object(Logger, "info")


@pytest.fixture
def mock_logger_typewriter_log(mocker):
    return mocker.patch.object(Logger, "typewriter_log")


@pytest.fixture
def mock_logger_warn(mocker):
    return mocker.patch.object(Logger, "warn")


@pytest.fixture
def mock_logger_error(mocker):
    return mocker.patch.object(Logger, "error")


@pytest.fixture
def mock_say_text(mocker):
    return mocker.patch("autogpt.agent.agent.say_text")


@pytest.fixture
def mock_say_text_in_logs(mocker):
    """Only needed to supress actual speech in tests"""
    return mocker.patch("autogpt.logs.say_text")


@pytest.fixture
def typewriter_first_9(mocker):
    return [
        mocker.call(
            "TEST AI THOUGHTS:",
            "\x1b[33m",
            "I will start by browsing the website www.steinhaug.no to gather information and save it to a file.",
        ),
        mocker.call(
            "REASONING:",
            "\x1b[33m",
            "The website is likely to have information about Steinhaug ut på eventyr, which is the topic of my "
            "report. Browsing the website is a good starting point for gathering information.",
        ),
        mocker.call("PLAN:", "\x1b[33m", ""),
        mocker.call("- ", "\x1b[32m", "Browse www.steinhaug.no"),
        mocker.call("- ", "\x1b[32m", "Save information to file"),
        mocker.call("- ", "\x1b[32m", "Repeat for YouTube channel"),
        mocker.call("- ", "\x1b[32m", "Write structured report"),
        mocker.call(
            "CRITICISM:",
            "\x1b[33m",
            "I need to ensure that I am not gathering irrelevant information from the website and that I am saving "
            "the information in a structured manner.",
        ),
        mocker.call(
            "NEXT ACTION: ",
            "\x1b[36m",
            "COMMAND = \x1b[36mbrowse_website\x1b[0m  ARGUMENTS = \x1b[36m{'url': 'www.steinhaug.no', 'question': "
            "'all information'}\x1b[0m",
        ),
    ]


@pytest.fixture
def typewriter_first_9_with_speak(mocker, typewriter_first_9):
    typewriter_first_9.insert(
        8,
        mocker.call(
            "SPEAK:",
            "\x1b[33m",
            "I will start by browsing the website www.steinhaug.no to gather information and save it to a file.",
        ),
    )
    return typewriter_first_9


@pytest.fixture
def typewriter_command_returned(mocker):
    return [
        mocker.call(
            "SYSTEM: ",
            "\x1b[33m",
            "Command browse_website returned: command return value",
        ),
    ]


@pytest.fixture
def typewriter_commands_left(mocker):
    return [
        mocker.call(
            "\x1b[36mAUTHORISED COMMANDS LEFT: \x1b[0m0",
        ),
    ]


@single_agent_loop_setup
def test_execute_command_continuous_mode(
    config,
    agent,
    mocker,
    mock_input,
    mock_chat_with_ai,
    mock_execute_command,
    mock_logger_info,
    mock_logger_typewriter_log,
    mock_logger_warn,
    mock_logger_error,
    mock_say_text,
    mock_say_text_in_logs,
    typewriter_first_9,
    typewriter_command_returned,
    typewriter_commands_left,
):
    # Test with browse_website expected outcome in continuous mode with limit = 1
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

    expected_typewriter_log = (
        typewriter_first_9
        + typewriter_commands_left
        + typewriter_command_returned
        + [
            mocker.call("Continuous Limit Reached: ", "\x1b[33m", "1"),
        ]
    )

    actual_logger_typewriter_log_calls = mock_logger_typewriter_log.call_args_list
    assert actual_logger_typewriter_log_calls == expected_typewriter_log

    expected_logger_info = []
    actual_logger_info_calls = mock_logger_info.call_args_list
    assert actual_logger_info_calls == expected_logger_info

    expected_logger_warn = []
    actual_logger_warn_calls = mock_logger_warn.call_args_list
    assert actual_logger_warn_calls == expected_logger_warn

    expected_logger_error = []
    actual_logger_error_calls = mock_logger_error.call_args_list
    assert actual_logger_error_calls == expected_logger_error

    expected_say_text = [mocker.call("I want to execute browse_website")]
    actual_say_text_calls = mock_say_text.call_args_list
    assert actual_say_text_calls == expected_say_text


@setup_with_chat
def test_chat_messages_enabled(config, agent, mocker, mock_chat_with_ai):
    mock_clean_input = mocker.patch("autogpt.agent.agent.clean_input")
    mock_clean_input.return_value = "n"
    agent.start_interaction_loop()

    expected_clean_input = [mocker.call("Waiting for your response...")]
    actual_clean_input_calls = mock_clean_input.call_args_list
    assert actual_clean_input_calls == expected_clean_input


# Test case when we were unable to parse the command from the JSON
def test_command_error(
    config,
    agent,
    mocker,
    mock_chat_with_ai,
    mock_input,
    mock_logger_typewriter_log,
    typewriter_first_9_with_speak,
    typewriter_command_auth_by_user,
):
    mock_get_command = mocker.patch("autogpt.agent.agent.get_command")
    mock_get_command.return_value = "Error:", "Missing 'command' object in JSON"
    mock_input.side_effect = ["y", "n"]

    agent.start_interaction_loop()

    typewriter_first_8 = typewriter_first_9_with_speak[:-1]

    expected_typewriter_log = (
        typewriter_first_8
        + [
            mocker.call(
                "NEXT ACTION: ",
                "\x1b[36m",
                "COMMAND = \x1b[36mError:\x1b[0m  ARGUMENTS = \x1b[36mMissing 'command' object in JSON\x1b[0m",
            )
        ]
        + typewriter_command_auth_by_user
        + [
            mocker.call(
                "SYSTEM: ",
                "\x1b[33m",
                "Command Error: threw the following error: Missing 'command' object in JSON",
            )
        ]
        + typewriter_first_8
        + [
            mocker.call(
                "NEXT ACTION: ",
                "\x1b[36m",
                "COMMAND = \x1b[36mError:\x1b[0m  ARGUMENTS = \x1b[36mMissing 'command' object in JSON\x1b[0m",
            )
        ]
    )

    actual_logger_typewriter_log_calls = mock_logger_typewriter_log.call_args_list
    assert actual_logger_typewriter_log_calls == expected_typewriter_log


@pytest.fixture
def typewriter_command_auth_by_user(mocker):
    return [
        mocker.call(
            "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=", "\x1b[35m", ""
        )
    ]


@pytest.fixture
def typewriter_verify_by_agent(mocker):
    return [
        mocker.call(
            "-=-=-=-=-=-=-= THOUGHTS, REASONING, PLAN AND CRITICISM WILL NOW BE VERIFIED BY AGENT -=-=-=-=-=-=-=",
            "\x1b[32m",
            "",
        )
    ]


@pytest.fixture
def self_feedback_test():
    return (
        "Y This plan effectively addresses the role of an AI agent tasked with gathering information "
        "about Steinhaug ut på eventyr and writing a structured report. The agent's thought process "
        "of browsing the website and YouTube channel is a logical starting point for gathering "
        "information. The reasoning of saving information to a file and repeating the process for the "
        "YouTube channel ensures that all relevant information is collected and organized. The plan "
        "also addresses the concern of gathering irrelevant information and saving it in a structured "
        "manner. Overall, this plan is effective in achieving the role's objectives."
    )


@pytest.fixture
def typewriter_self_feedback(mocker, self_feedback_test):
    return [
        mocker.call(
            f"SELF FEEDBACK: {self_feedback_test}",
            "\x1b[33m",
            "",
        )
    ]


@pytest.fixture
def typewriter_human_feedback(mocker, self_feedback_test):
    return [mocker.call("SYSTEM: ", "\x1b[33m", "Human feedback: some human feedback")]


@pytest.fixture
def mock_get_self_feedback(mocker, agent, self_feedback_test):
    mock = mocker.patch.object(agent, "get_self_feedback")
    mock.return_value = self_feedback_test
    return mock


@pytest.fixture
def info_action_prompt(mocker):
    return [
        mocker.call(
            "Enter 'y' to authorise command, 'y -N' to run N continuous commands, 's' to run self-feedback "
            "commands'n' to exit program, or enter feedback for Test AI...",
        )
    ]


@pytest.fixture
def info_asking_user(mocker):
    return [
        mocker.call(
            "Asking user via keyboard...",
        )
    ]


@pytest.fixture
def info_exiting(mocker):
    return [mocker.call("Exiting...")]


def test_execute_command_normal_mode(
    agent,
    mocker,
    config,
    mock_input,
    mock_chat_with_ai,
    mock_execute_command,
    mock_logger_info,
    mock_logger_typewriter_log,
    mock_logger_warn,
    mock_logger_error,
    mock_say_text,
    mock_get_self_feedback,
    typewriter_first_9_with_speak,
    typewriter_command_auth_by_user,
    typewriter_command_returned,
    typewriter_verify_by_agent,
    typewriter_self_feedback,
    typewriter_human_feedback,
    info_action_prompt,
    info_asking_user,
    info_exiting,
):
    mock_input.side_effect = ["", "y", "s", "y - 3", "some human feedback", "y -3", "n"]
    agent.start_interaction_loop()

    expected_typewriter_log = (
        typewriter_first_9_with_speak
        + typewriter_command_auth_by_user
        + typewriter_command_returned
        + typewriter_first_9_with_speak
        + typewriter_verify_by_agent
        + typewriter_self_feedback
        + typewriter_command_auth_by_user
        + typewriter_command_returned
        + typewriter_first_9_with_speak
        + typewriter_human_feedback
        + typewriter_first_9_with_speak
        + typewriter_command_auth_by_user
        + typewriter_command_returned
        + typewriter_first_9_with_speak
        + typewriter_command_returned
        + typewriter_first_9_with_speak
        + typewriter_command_returned
        + typewriter_first_9_with_speak
    )

    actual_logger_typewriter_log_calls = mock_logger_typewriter_log.call_args_list
    assert actual_logger_typewriter_log_calls == expected_typewriter_log

    expected_logger_info = (
        info_action_prompt
        + info_asking_user * 2
        + info_action_prompt
        + info_asking_user
        + info_action_prompt
        + info_asking_user
        + info_asking_user
        + info_action_prompt
        + info_asking_user
        + info_action_prompt
        + info_asking_user
        + info_exiting
    )

    actual_logger_info_calls = mock_logger_info.call_args_list
    assert actual_logger_info_calls == expected_logger_info

    expected_logger_warn = [
        mocker.call("Invalid input format."),
        mocker.call(
            "Invalid input format. Please enter 'y -n' where n is the number of continuous tasks."
        ),
    ]

    actual_logger_warn_calls = mock_logger_warn.call_args_list
    assert actual_logger_warn_calls == expected_logger_warn

    expected_logger_error = []
    actual_logger_error_calls = mock_logger_error.call_args_list
    assert actual_logger_error_calls == expected_logger_error


def test_execute_command_no_next_command_self_feedback(
    agent,
    mocker,
    config,
    mock_input,
    mock_chat_with_ai,
    mock_execute_command,
    mock_logger_info,
    mock_logger_typewriter_log,
    mock_logger_warn,
    mock_logger_error,
    mock_say_text,
    mock_get_self_feedback,
    typewriter_first_9_with_speak,
    typewriter_command_auth_by_user,
    typewriter_command_returned,
    typewriter_verify_by_agent,
    typewriter_self_feedback,
    typewriter_human_feedback,
    info_action_prompt,
    info_asking_user,
    info_exiting,
):
    mock_input.side_effect = ["s", "n"]
    self_feedback_test = "Will not generate next command"
    mock_get_self_feedback.return_value = self_feedback_test
    agent.start_interaction_loop()

    expected_typewriter_log = (
        typewriter_first_9_with_speak
        + typewriter_verify_by_agent
        + [
            mocker.call(
                f"SELF FEEDBACK: {self_feedback_test}",
                "\x1b[33m",
                "",
            )
        ]
        + typewriter_command_returned
        + typewriter_first_9_with_speak
    )

    actual_logger_typewriter_log_calls = mock_logger_typewriter_log.call_args_list
    assert actual_logger_typewriter_log_calls == expected_typewriter_log

    expected_logger_info = (
        info_action_prompt
        + info_asking_user
        + info_action_prompt
        + info_asking_user
        + info_exiting
    )

    actual_logger_info_calls = mock_logger_info.call_args_list
    assert actual_logger_info_calls == expected_logger_info


@pytest.fixture
def expected_logger_error(mocker):
    return [mocker.call("Error: \n", "The Exception")]


# Test error handling for print_assistant_thoughts
@single_agent_loop_setup
def test_execute_command_print_assistant_thoughts_error(
    config,
    agent,
    mocker,
    mock_input,
    mock_logger_error,
    mock_chat_with_ai,
    mock_execute_command,
    mock_say_text,
    mock_say_text_in_logs,
    expected_logger_error,
):
    mock_print_assistant_thoughts = mocker.patch(
        "autogpt.agent.agent.print_assistant_thoughts"
    )
    mock_print_assistant_thoughts.side_effect = Exception("The Exception")

    agent.start_interaction_loop()

    actual_logger_error_calls = mock_logger_error.call_args_list
    assert actual_logger_error_calls == expected_logger_error


# Test error handling for say_text
@single_agent_loop_setup
def test_say_text_error(
    config,
    agent,
    mock_input,
    mock_logger_error,
    mock_chat_with_ai,
    mock_execute_command,
    mock_say_text,
    mock_say_text_in_logs,
    expected_logger_error,
):
    mock_say_text.side_effect = Exception("The Exception")

    agent.start_interaction_loop()

    actual_logger_error_calls = mock_logger_error.call_args_list
    assert actual_logger_error_calls == expected_logger_error


# Test error handling for _resolve_pathlike_command_args
@single_agent_loop_setup
def test_resolve_pathlike_command_args_error(
    config,
    agent,
    mocker,
    mock_input,
    mock_logger_error,
    mock_chat_with_ai,
    mock_execute_command,
    mock_say_text,
    mock_say_text_in_logs,
    expected_logger_error,
):
    mock_resolve_pathlike_command_args = mocker.patch.object(
        agent, "_resolve_pathlike_command_args"
    )
    mock_resolve_pathlike_command_args.side_effect = Exception("The Exception")

    agent.start_interaction_loop()

    actual_logger_error_calls = mock_logger_error.call_args_list
    assert actual_logger_error_calls == expected_logger_error


# Test error handling for get_command
@single_agent_loop_setup
def test_execute_command_get_command_error(
    config,
    agent,
    mocker,
    mock_input,
    mock_logger_error,
    mock_chat_with_ai,
    mock_execute_command,
    mock_say_text,
    mock_say_text_in_logs,
    expected_logger_error,
):
    mock_get_command = mocker.patch("autogpt.agent.agent.get_command")
    mock_get_command.side_effect = Exception("The Exception")

    agent.start_interaction_loop()

    actual_logger_error_calls = mock_logger_error.call_args_list
    assert actual_logger_error_calls == expected_logger_error
