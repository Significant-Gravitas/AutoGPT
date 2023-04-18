"""Agent module."""
import click

from autogpt.agent.agent import Agent
from autogpt.agent.agent_manager import AgentManager

__all__ = ["Agent", "AgentManager"]


@click.group()
def agent() -> None:
    """Commands for running an Auto-GPT agent."""
    pass


@agent.command()
def v2() -> None:
    """Command for running the v2 Auto-GPT agent."""
    import trio

    import autogpt.agent.autogpt_core

    # run the agent
    trio.run(AgentManager.run, autogpt.agent.autogpt_core.AutoGptCore)
