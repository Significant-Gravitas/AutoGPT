"""
Some GPT-4 Generated Message Objects for testing agent messaging.
"""
from typing import Any

from autogpt.core.schema import BaseMessage


class ShutdownMessage(BaseMessage):
    """A message that tells the agent to shutdown for some information."""
    immediately: bool


class QueryMessage(BaseMessage):
    """A message that queries for some information."""

    query: str


class CommandMessage(BaseMessage):
    """A message that commands the recipient to perform some action."""

    command: str
    parameters: dict[str, Any] = {}


class ResponseMessage(BaseMessage):
    """A message that contains the response to a query or command."""

    response: Any
    success: bool = True
    error_message: str = ""


class StatusMessage(BaseMessage):
    """A message that reports the status of the sender."""

    status: str


class InfoMessage(BaseMessage):
    """A message that provides some information to the recipient."""

    info: str
