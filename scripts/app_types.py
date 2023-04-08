from pydantic import BaseModel
from typing import List, Literal


class StartBody(BaseModel):
    ai_name: str
    ai_role: str
    ai_goals: List[str]


class ChatBody(BaseModel):
    message: str | Literal['y'] | Literal['n']


class ConsoleMessage(BaseModel):
    title: str
    message: str | None
