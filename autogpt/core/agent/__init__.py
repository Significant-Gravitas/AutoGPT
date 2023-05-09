"""The Agent is an autonomouos entity guided by a LLM provider."""
from autogpt.core.agent.base import Agent
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.INTERFACE_DONE,
    handoff_notes="Interface has been created. Work is needed on the agent factory.",
)
