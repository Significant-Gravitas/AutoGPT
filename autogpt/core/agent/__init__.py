"""The Agent is an autonomouos entity guided by a LLM provider."""
from autogpt.core.agent.base import Agent
from autogpt.core.status import Status

status = Status.INTERFACE_DONE
handover_notes = "Interface has been created. Work is needed on the agent factory."
