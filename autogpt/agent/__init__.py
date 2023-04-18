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
    pass
