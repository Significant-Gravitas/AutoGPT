from pydantic import BaseModel


class Task(BaseModel):
    """Jsonifiable representation of a task"""

    user_input: str
