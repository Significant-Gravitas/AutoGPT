from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from autogpt.agent import Agent
from autogpt.config import Config


@pytest.fixture
def agent():
    ai_name = "Test AI"
    memory = MagicMock()
    full_message_history = []
    autonomous_cycles_budget = 0
    command_registry = MagicMock()
    config = Config()
    system_prompt = "System prompt"
    triggering_prompt = "Triggering prompt"
    workspace_directory = "workspace_directory"

    agent = Agent(
        ai_name,
        memory,
        full_message_history,
        autonomous_cycles_budget,
        command_registry,
        config,
        system_prompt,
        triggering_prompt,
        workspace_directory,
    )
    return agent


def test_agent_initialization(config: Config, agent: Agent):
    assert agent.ai_name == "Test AI"
    assert agent.memory == agent.memory
    assert agent.full_message_history == []
    assert agent.autonomous_cycles_remaining == 0
    assert agent.command_registry == agent.command_registry
    assert agent.config == agent.config
    assert agent.system_prompt == "System prompt"
    assert agent.triggering_prompt == "Triggering prompt"


def test_should_prompt_user(config: Config, agent: Agent, mocker: MockerFixture):
    mocker.patch.object(config, 'continuous_mode', False)
    assert agent.should_prompt_user is True

    mocker.patch.object(agent, 'autonomous_cycles_remaining', 2)
    assert agent.should_prompt_user is False

    mocker.patch.object(config, 'continuous_mode', True)
    mocker.patch.object(agent, 'autonomous_cycles_remaining', 0)
    assert agent.should_prompt_user is False

    mocker.patch.object(agent, 'autonomous_cycles_remaining', 100)
    assert agent.should_prompt_user is False


def test_user_feedback_prompt(agent: Agent, mocker: MockerFixture):
    mocker.patch.object(agent, 'ai_name', 'the_ai_name')
    expected_message = (
        "Enter 'y' to authorise command, 'y -N' to run N continuous commands, "
        "'s' to run self-feedback commands, "
        "'n' to exit program, or enter feedback for the_ai_name... "
    )
    assert agent.user_feedback_prompt == expected_message


def test_determine_next_command_from_user_input():
    # user entered 'y' -> generate next command
    command_name, remaining_cycles, user_input = Agent.determine_next_command(
        "y"
    )
    assert command_name is None
    assert remaining_cycles == 0
    assert user_input == "GENERATE NEXT COMMAND JSON"

    # user entered 'y -N' -> generate next command with N remaining cycles
    command_name, remaining_cycles, user_input = Agent.determine_next_command(
        "y -20"
    )
    assert command_name is None
    assert remaining_cycles == 20
    assert user_input == "GENERATE NEXT COMMAND JSON"

    # user entered 'y -[invalid number]' -> print error
    command_name, remaining_cycles, user_input = Agent.determine_next_command(
        "y -X"
    )
    assert command_name == "input error"
    assert remaining_cycles == 0
    assert (
        user_input
        == "Invalid input format. Please enter 'y -n' where n is the number of continuous tasks."
    )

    # user entered 'n' -> exit
    command_name, remaining_cycles, user_input = Agent.determine_next_command(
        "n"
    )
    assert command_name is None
    assert remaining_cycles == 0
    assert user_input == "EXIT"

    # user entered text -> human_feedback
    the_input = "Some item to tell the AI about while it is doing stuff."
    command_name, remaining_cycles, user_input = Agent.determine_next_command(
        the_input
    )
    assert command_name == "human_feedback"
    assert remaining_cycles == 0
    assert user_input == the_input

    # user entered 's' -> self_feedback
    the_input = "s"
    command_name, remaining_cycles, user_input = Agent.determine_next_command(
        the_input
    )
    assert command_name == "self_feedback"
    assert remaining_cycles == 0
    assert user_input is None

    # empty input -> print error
    the_input = "      "
    command_name, remaining_cycles, user_input = Agent.determine_next_command(
        the_input
    )
    assert command_name == "input error"
    assert remaining_cycles == 0
    assert user_input == "Invalid input format."
