"""Messages that the agent can send and receive."""
import enum

import pydantic


class AgentCommands(enum.Enum):
    """The commands that the agent can send to the AI."""

    ping = "ping"
    shutdown = "shutdown"
    test = "test"


class Command(pydantic.BaseModel):
    """A command that the AI can execute."""

    ai_core_id: str
    command: str
    arguments: dict[str, str]


class Event(pydantic.BaseModel):
    """A event that the AI can receive."""

    ai_core_id: str
    event: str
    data: dict[str, str]
