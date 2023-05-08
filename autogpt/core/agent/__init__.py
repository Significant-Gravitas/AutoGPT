"""The Agent is an autonomouos entity guided by a LLM provider."""
from autogpt.core.agent.base import Agent
import autogpt.core.status

status = autogpt.core.status.Status.INTERFACE_DONE
handover_notes = "Interface has been created. Work is needed on the agent factory."
