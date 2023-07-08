from autogpt.agent import Agent
from autogpt.app import execute_command


def check_plan():
    return "hi"


def test_execute_command_plugin(agent: Agent):
    """Test that executing a command that came from a plugin works as expected"""
    command_name = "check_plan"
    agent.ai_config.prompt_generator.add_command(
        command_name,
        "Read the plan.md with the next goals to achieve",
        {},
        check_plan,
    )
    command_result = execute_command(
        command_name=command_name,
        arguments={},
        agent=agent,
    )
    assert command_result == "hi"
