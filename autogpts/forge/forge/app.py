import os

from forge.agent import ForgeAgent
from forge.sdk import AgentDB, LocalWorkspace

database_name = os.getenv("DATABASE_STRING")
workspace = LocalWorkspace(os.getenv("AGENT_WORKSPACE"))
database = AgentDB(database_name, debug_enabled=False)
agent = ForgeAgent(database=database, workspace=workspace)

app = agent.get_agent_app()
