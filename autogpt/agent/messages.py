"""Messages that the agent can send and receive."""
import pydantic


class Command(pydantic.BaseModel):
    """A command that the AI can execute."""

    ai_core_id: str
    name: str
    arguments: dict[str, str]


class Event(pydantic.BaseModel):
    """A event that the AI can receive."""

    ai_core_id: str
    name: str
    data: dict[str, str]
