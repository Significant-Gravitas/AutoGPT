from __future__ import annotations
import abc
import enum
from typing import ClassVar, Optional, Literal
from pydantic import BaseModel
from AFAAS.lib.sdk.logger import AFAASLogger
from langchain_core.messages import ChatMessage, HumanMessage, SystemMessage, AIMessage, FunctionMessage

LOG = AFAASLogger(name=__name__)

class AbstractRoleLabels(abc.ABC, BaseModel):
    USER: str
    SYSTEM: str
    ASSISTANT: str

    FUNCTION: Optional[str] = None
    """May be used for the return value of function calls"""

class Role(str, enum.Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"

    FUNCTION = "function"

class OpenAIRoleLabel(AbstractRoleLabels):
    USER : str = "user"
    SYSTEM : str = "system"
    ASSISTANT : str = "assistant"

    FUNCTION : str = "function"
    """May be used for the return value of function calls"""


class AbstractChatMessage(abc.ABC, BaseModel):
    _role_labels: ClassVar[AbstractRoleLabels]
    role: str
    content: str

    @classmethod
    def assistant(cls, content: str) -> "AbstractChatMessage":
        return cls(role=cls._role_labels.ASSISTANT, content=content)

    @classmethod
    def user(cls, content: str) -> "AbstractChatMessage":
        return cls(role=cls._role_labels.USER, content=content)

    @classmethod
    def system(cls, content: str) -> "AbstractChatMessage":
        return cls(role=cls._role_labels.SYSTEM, content=content)


    def to_langchain(self) -> ChatMessage:
        if self.role == self._role_labels.ASSISTANT:
            return AIMessage(content=self.content)
        elif self.role == self._role_labels.USER:
            return HumanMessage(content=self.content)
        elif self.role == self._role_labels.SYSTEM:
            return SystemMessage(content=self.content)
        elif self.role == self._role_labels.FUNCTION:
            return FunctionMessage(content=self.content)
        else:
            raise ValueError(f"Unknown role: {self.role}")

    @classmethod
    def from_langchain(cls, message: ChatMessage) -> "AbstractChatMessage":
        if isinstance(message, AIMessage):
            return cls.assistant(content=message.content)
        elif isinstance(message, HumanMessage):
            return cls.user(content=message.content)
        elif isinstance(message, SystemMessage):
            return cls.system(content=message.content)
        elif isinstance(message, FunctionMessage):
            return cls.system(content=message.content)
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

    def dict(self, **kwargs):
        d = super().dict(**kwargs)
        d["role"] = self.role
        return d

class AFAASChatMessage(BaseModel):
    role: Role
    content: str

    @staticmethod
    def assistant(content: str) -> "AFAASChatMessage":
        LOG.warning("AFAASChatMessage is deprecated. Use OpenAIChatMessage or Langchain.AIMessage instead.")
        return AFAASChatMessage(role=Role.ASSISTANT, content=content)

    @staticmethod
    def user(content: str) -> "AFAASChatMessage":
        LOG.warning("AFAASChatMessage is deprecated. Use OpenAIChatMessage or Langchain.HumanMessage instead.")
        return AFAASChatMessage(role=Role.USER, content=content)

    @staticmethod
    def system(content: str) -> "AFAASChatMessage":
        LOG.warning("AFAASChatMessage.assistant is deprecated. Use OpenAIChatMessage or Langchain.SystemMessage instead.")
        return AFAASChatMessage(role=Role.SYSTEM, content=content)

    def dict(self, **kwargs):
        d = super().dict(**kwargs)
        d["role"] = self.role.value
        return d

class OpenAIChatMessage(AbstractChatMessage):
    _role_labels: ClassVar[OpenAIRoleLabel] = OpenAIRoleLabel()


class AssistantFunctionCall(BaseModel):
    name: str
    arguments: str

class AssistantToolCall(BaseModel):
    # id: str
    type: Literal["function"]
    function: AssistantFunctionCall

class AssistantChatMessage(AFAASChatMessage):

    role: Role.ASSISTANT
    content: Optional[str] = None
    tool_calls: Optional[list[AssistantToolCall]] = None
