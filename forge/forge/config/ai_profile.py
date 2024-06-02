from pydantic import BaseModel, Field


class AIProfile(BaseModel):
    """
    Object to hold the AI's personality.

    Attributes:
        ai_name (str): The name of the AI.
        ai_role (str): The description of the AI's role.
        ai_goals (list): The list of objectives the AI is supposed to complete.
        api_budget (float): The maximum dollar value for API calls (0.0 means infinite)
    """

    ai_name: str = ""
    ai_role: str = ""
    ai_goals: list[str] = Field(default_factory=list[str])
    api_budget: float = 0.0
