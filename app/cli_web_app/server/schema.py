from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field, validator


class AgentInfo(BaseModel):
    id: UUID = None
    objective: str = ""
    name: str = ""
    role: str = ""
    goals: list[str] = []


class AgentConfiguration(BaseModel):
    """Configuration for creation of a new agent."""

    # We'll want to get this schema from the configuration, so it needs to be dynamic.
    user_configuration: dict
    agent_goals: AgentInfo

    @validator("agent_goals")
    def only_objective_or_name_role_goals(cls, agent_goals):
        goals_specification = [agent_goals.name, agent_goals.role, agent_goals.goals]
        if agent_goals.objective and any(goals_specification):
            raise ValueError("Cannot specify both objective and name, role, or goals")
        if not agent_goals.objective and not all(goals_specification):
            raise ValueError("Must specify either objective or name, role, and goals")


class AgentMessageRequestBody(BaseModel):
    message: str = Field(..., min_length=1)
    start: bool = Field(default=False)


class PlannerAgentMessageResponseBody(BaseModel):
    ability_result: Dict[str, Any]
    current_task: Optional[Any]
    next_ability: Optional[Any]
