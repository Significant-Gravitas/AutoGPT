from __future__ import annotations

import uuid

from AFAAS.core.agents.planner.main import PlannerAgent
from AFAAS.interfaces.db.db import AbstractMemory, MemoryAdapterType, MemoryConfig
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)

user_id = "pytest_U3ba0a1c6-8cdf-4daa-a244-297b2057146a"

agent_settings: PlannerAgent.SystemSettings = PlannerAgent.SystemSettings(
    user_id=user_id,
    agent_id="pytest_A639f7cda-c88c-44d7-b0b2-a4a4abbd4a6c" + str(uuid.uuid4()),
    agent_goal_sentence="Prepare a family dinner",
)

memory_config = MemoryConfig()
memory_config.json_file_path += "/pytest"
memory_settings = AbstractMemory.SystemSettings()
memory_settings.configuration = memory_config

agent = PlannerAgent(
    settings=agent_settings,
    **agent_settings.dict(),
    memory=AbstractMemory.get_adapter(settings=memory_settings),
)

PLANNERAGENT = agent


# import sys
# from pathlib import Path

# # Get the parent directory of the current file (conftest.py)
# parent_dir = Path(__file__).parent
# sys.path.append(str(parent_dir))


def agent_dataset() -> PlannerAgent:
    import uuid

    from AFAAS.core.agents.planner.main import PlannerAgent
    from AFAAS.interfaces.db.db import AbstractMemory, MemoryAdapterType, MemoryConfig
    from AFAAS.lib.sdk.logger import AFAASLogger

    LOG = AFAASLogger(name=__name__)

    user_id = "pytest_U3ba0a1c6-8cdf-4daa-a244-297b2057146a"

    agent_settings: PlannerAgent.SystemSettings = PlannerAgent.SystemSettings(
        user_id=user_id,
        agent_id="pytest_A639f7cda-c88c-44d7-b0b2-a4a4abbd4a6c",
        agent_goal_sentence="Prepare a family dinner",
    )

    agent = PlannerAgent(settings=agent_settings, **agent_settings.dict())
    memory_config = MemoryConfig()
    memory_config.json_file_path += "/pytest"
    memory_settings = AbstractMemory.SystemSettings()
    memory_settings.configuration = memory_config

    agent = PlannerAgent(
        settings=agent_settings,
        **agent_settings.dict(),
        memory=AbstractMemory.get_adapter(settings=memory_settings),
    )
    return agent
