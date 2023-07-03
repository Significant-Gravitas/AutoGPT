"""The Agent is an autonomouos entity guided by a LLM provider."""
from autogpt.core.agent.base import Agent
from autogpt.core.agent.simple import AgentSettings, SimpleAgent
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.INTERFACE_DONE,
    handoff_notes=(
        "Before times: Sketched out Agent.__init__\n"
        "5/7: Interface has been created. Work is needed on the agent factory.\n"
        "5/8: Agent factory has been adjusted to use the new plugin system."
        "5/15: Get configuration compilation working.\n"
        "5/16: Agent can boot strap and stand up. Working on taking an execution step.\n"
    ),
)
