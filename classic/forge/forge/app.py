import os
from pathlib import Path

from forge.agent.forge_agent import ForgeAgent
from forge.agent_protocol.database.db import AgentDB
from forge.file_storage import FileStorageBackendName, get_storage
from forge.llm.providers import MultiProvider

database_name = os.getenv("DATABASE_STRING")
workspace = get_storage(FileStorageBackendName.LOCAL, root_path=Path("workspace"))
database = AgentDB(database_name, debug_enabled=False)
llm_provider = MultiProvider()
agent = ForgeAgent(database=database, workspace=workspace, llm_provider=llm_provider)

app = agent.get_agent_app()
