# models/interaction.py
from typing import Optional

from pydantic import BaseModel


class Interaction(BaseModel):
    user_input: str
    agent_output: Optional[str] = None
