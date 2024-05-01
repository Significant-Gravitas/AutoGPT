import os

from forge.agent import ForgeAgent
from forge.file_storage import FileStorageBackendName, get_storage

from .sdk import AgentDB

database_name = os.getenv("DATABASE_STRING")
file_storage = get_storage(FileStorageBackendName.LOCAL)
database = AgentDB(database_name, debug_enabled=False)
agent = ForgeAgent(database=database, workspace=file_storage)

app = agent.get_agent_app()
