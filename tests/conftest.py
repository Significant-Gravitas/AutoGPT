import uuid
from pathlib import Path

import pytest
from AFAAS.core.agents.planner.main import PlannerAgent
from AFAAS.core.workspace import AbstractFileWorkspace
from AFAAS.interfaces.tools.base import BaseToolsRegistry


def agent_dataset(
) -> PlannerAgent:
    PlannerAgentSettings = PlannerAgent.SystemSettings(user_id= 'pytest_U' + str(uuid.uuid4()) ,
                                                       agent_goal_sentence = 'Make a plan to build a poultry house',
                                                    )
    agent = PlannerAgent(
        settings= PlannerAgentSettings,
        **PlannerAgentSettings.dict()
    )
    return agent

@pytest.fixture
def agent() -> PlannerAgent:
    return agent_dataset()

@pytest.fixture
def local_workspace(
) -> AbstractFileWorkspace : 
    return agent_dataset().workspace

@pytest.fixture
def empty_tool_registry() -> BaseToolsRegistry:
    registry = agent_dataset().tool_registry
    registry.tools = {}
    return registry
