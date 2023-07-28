from autogpt.agents.agent import Agent, execute_command


def test_agent_initialization(agent: Agent):
    assert agent.ai_config.ai_name == "Base"
    assert agent.history.messages == []
    assert agent.cycle_budget is None
    assert "You are Base" in agent.system_prompt


def test_execute_command_plugin(agent: Agent):
    """Test that executing a command that came from a plugin works as expected"""
    command_name = "check_plan"
    agent.ai_config.prompt_generator.add_command(
        command_name,
        "Read the plan.md with the next goals to achieve",
        {},
        lambda: "hi",
    )
    command_result = execute_command(
        command_name=command_name,
        arguments={},
        agent=agent,
    )
    assert command_result == "hi"


# More test methods can be added for specific agent interactions
# For example, mocking chat_with_ai and testing the agent's interaction loop
