from typing import Optional

from pydantic import BaseModel


class AgentBase(BaseModel):
    name: str
    description: Optional[str]


class AgentCreate(AgentBase):
    pass


class AgentUpdate(AgentBase):
    pass


class Agent(AgentBase):
    id: int

    class Config:
        orm_mode = True
