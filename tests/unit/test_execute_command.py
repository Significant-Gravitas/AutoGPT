from autogpt.agent import Agent
from autogpt.app import execute_command


def check_plan():
    return "hi"


def test_execute_command_plugin(agent: Agent):
    """Test that executing a command that came from a plugin works as expected"""
    agent.ai_config.prompt_generator.add_command(
        "check_plan",
        "Read the plan.md with the next goals to achieve",
        {},
        check_plan,
    )
    command_name = "check_plan"
    arguments = {}
    command_result = execute_command(
        command_name=command_name,
        arguments=arguments,
        agent=agent,
    )
    assert command_result == "hi"
