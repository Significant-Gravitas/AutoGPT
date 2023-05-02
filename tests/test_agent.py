from unittest.mock import MagicMock

import pytest

from autogpt.agent import Agent
from autogpt.config import Config


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


# More test methods can be added for specific agent interactions
# For example, mocking chat_with_ai and testing the agent's interaction loop

def test_should_prompt_user_given_not_in_continuous_mode_and_next_action_count_is_zero_returns_true(agent):
    agent.next_action_count = 0
    assert agent.should_prompt_user(False) is True


def test_should_prompt_user_given_not_in_continuous_mode_and_next_action_count_is_greater_than_zero_return_false(agent):
    agent.next_action_count = 2
    assert agent.should_prompt_user(False) is False


def test_should_prompt_user_given_in_continuous_mode_and_next_action_count_is_zero_returns_false(agent):
    agent.next_action_count = 0
    assert agent.should_prompt_user(True) is False


def test_should_prompt_user_given_in_continuous_mode_and_next_action_count_is_greater_than_zero_returns_false(agent):
    agent.next_action_count = 100
    assert agent.should_prompt_user(True) is False


def test_generate_user_feedback_message_should_format_message_with_ai_name(agent):
    agent.ai_name = "the_ai_name"
    expected_message = "Enter 'y' to authorise command, 'y -N' to run N continuous commands, " \
                       "'s' to run self-feedback commands, " \
                       "'n' to exit program, or enter feedback for the_ai_name... "
    assert agent.generate_user_feedback_message() == expected_message


def test_calculate_next_command_from_user_input_given_user_entered_y_result_is_generate_next_command(agent):
    command_name, next_count, user_input = agent.calculate_next_command_from_user_input("y")
    assert command_name is None
    assert next_count == 0
    assert user_input == "GENERATE NEXT COMMAND JSON"


def test_calculate_next_command_from_user_input_given_user_entered_y_dash_number_result_generate_next_command(agent):
    command_name, next_count, user_input = agent.calculate_next_command_from_user_input("y -20")
    assert command_name is None
    assert next_count == 20
    assert user_input == "GENERATE NEXT COMMAND JSON"


def test_calculate_next_command_from_user_input_given_user_entered_y_dash_and_invalid_number_print_error(agent):
    command_name, next_count, user_input = agent.calculate_next_command_from_user_input("y -X")
    assert command_name == "input error"
    assert next_count == 0
    assert user_input == "Invalid input format. Please enter 'y -n' where n is the number of continuous tasks."


def test_calculate_next_command_from_user_input_given_user_entered_n_result_is_exit(agent):
    command_name, next_count, user_input = agent.calculate_next_command_from_user_input("n")
    assert command_name is None
    assert next_count == 0
    assert user_input == "EXIT"


def test_calculate_next_command_from_user_input_given_user_text_result_is_feedback(agent):
    the_input = "Some item to tell the AI about while it is doing stuff."
    command_name, next_count, user_input = agent.calculate_next_command_from_user_input(the_input)
    assert command_name == "human_feedback"
    assert next_count == 0
    assert user_input == the_input


def test_calculate_next_command_from_user_input_given_s_turn_on_verify_feedback(agent):
    the_input = "s"
    command_name, next_count, user_input = agent.calculate_next_command_from_user_input(the_input)
    assert command_name == "self_feedback"
    assert next_count == 0
    assert user_input is None


def test_calculate_next_command_from_user_input_given_no_value_print_error(agent):
    the_input = "      "
    command_name, next_count, user_input = agent.calculate_next_command_from_user_input(the_input)
    assert command_name == "input error"
    assert next_count == 0
    assert user_input == "Invalid input format."
