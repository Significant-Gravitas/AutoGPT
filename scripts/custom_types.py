from enum import Enum
from typing import TypedDict

class RolesEnum(str, Enum):
    """The role of the message sender."""
    user: str = "user"
    assistant: str = "assistant"
    system: str = "system"


class ChatMessage(TypedDict):
    """A message added to the history of a chat session."""
    role: RolesEnum
    content: str
