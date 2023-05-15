# models/session.py
from typing import Optional

from pydantic import BaseModel


class SessionBase(BaseModel):
    agent_id: str


class SessionCreate(SessionBase):
    pass


class SessionUpdate(SessionBase):
    session_data: Optional[str] = None
