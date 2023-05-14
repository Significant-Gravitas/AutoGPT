"""The Agent is an autonomouos entity guided by a LLM provider."""
from autogpt.core.agent.base import Agent
from autogpt.core.agent.simple import SimpleAgent
from autogpt.core.agent.factory import SimpleAgentFactory
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.INTERFACE_DONE,
    handoff_notes=(
        "Before times: Sketched out Agent.__init__\n"
        "5/7: Interface has been created. Work is needed on the agent factory.\n"
        "5/8: Agent factory has been adjusted to use the new plugin system."
    ),
)
