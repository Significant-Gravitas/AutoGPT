"""Agent module."""
import asyncclick as click

from autogpt.agent.agent import Agent
from autogpt.agent.agent_manager import AgentManager

__all__ = ["Agent", "AgentManager"]


@click.group()
def agent() -> None:
    """Commands for running an Auto-GPT agent."""
    pass


@agent.command()
async def v2() -> None:
    """Command for running the v2 Auto-GPT agent."""
    import tracemalloc

    import trio

    import autogpt.agent.agent_manager

    trio.run(autogpt.agent.agent_manager.async_agent_manager)
