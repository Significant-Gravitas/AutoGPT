from typing import Type
from unittest.mock import MagicMock

import pytest
from _pytest.python_api import E as SomeError
from pytest_mock import MockerFixture

from autogpt.agent import Agent
from autogpt.config import AIConfig, Config


@pytest.fixture
def agent():
    return Agent(
        ai_name="Test AI",
        memory=MagicMock(),
        full_message_history=[],
        autonomous_cycles_budget=0,
        command_registry=MagicMock(),
        config=AIConfig(),
        system_prompt="System prompt",
        initial_prompt="Triggering prompt",
        workspace_directory="workspace_directory",
    )


def test_agent_initialization(config: Config, agent: Agent):
    assert agent.ai_name == "Test AI"
    assert agent.memory == agent.memory
    assert agent.full_message_history == []
    assert agent.autonomous_cycles_remaining == 0
    assert agent.command_registry == agent.command_registry
    assert agent.config == agent.config
    assert agent.system_prompt == "System prompt"
    assert agent.initial_prompt == "Triggering prompt"


def test_should_prompt_user(config: Config, agent: Agent, mocker: MockerFixture):
    mocker.patch.object(config, "continuous_mode", False)
    assert agent.should_prompt_user is True

    mocker.patch.object(agent, "autonomous_cycles_remaining", 2)
    assert agent.should_prompt_user is False

    mocker.patch.object(config, "continuous_mode", True)
    mocker.patch.object(agent, "autonomous_cycles_remaining", 0)
    assert agent.should_prompt_user is False

    mocker.patch.object(agent, "autonomous_cycles_remaining", 100)
    assert agent.should_prompt_user is False


def test_user_feedback_prompt(agent: Agent, mocker: MockerFixture):
    mocker.patch.object(agent, "ai_name", "the_ai_name")
    expected_message = (
        "Enter 'y' to authorise command, 'y -N' to run N continuous commands, "
        "'s' to run self-feedback commands, "
        "'n' to exit program, or enter feedback for the_ai_name... "
    )
    assert agent.user_feedback_prompt == expected_message


@pytest.mark.parametrize(
    "input, expected_output, expected_raise",
    [
        ("y", (None, 0, "GENERATE NEXT COMMAND JSON"), None),
        ("y -20", (None, 20, "GENERATE NEXT COMMAND JSON"), None),
        (
            "y -X",
            None,
            (
                SyntaxError,
                "Invalid input format. Please enter 'y -N' where N is the number of continuous tasks.",
            ),
        ),
        ("n", (None, 0, "EXIT"), None),
        ("s", ("self_feedback", 0, ""), None),
        ("      ", None, (ValueError, "Invalid input")),
        (
            "Test-feedback for the AI",
            ("human_feedback", 0, "Test-feedback for the AI"),
            None,
        ),
    ],
)
def test_determine_next_command_from_user_input(
    input,
    expected_output: tuple[str | None, int, str] | None,
    expected_raise: tuple[Type[SomeError], str] | None,
):
    if expected_raise:
        with pytest.raises(expected_raise[0], match=expected_raise[1]):
            Agent.determine_next_command(input)
        return

    if expected_output:
        output = Agent.determine_next_command(input)
        assert output == expected_output
