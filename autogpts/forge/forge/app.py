import os

import forge.sdk.db
from forge.sdk.workspace import LocalWorkspace
from forge.agent import Agent
import forge.sdk.forge_log
from dotenv import load_dotenv




def get_app():
    """Runs the agent server"""

    load_dotenv()
    forge.sdk.forge_log.setup_logger()

    database_name = os.getenv("DATABASE_STRING")
    workspace = LocalWorkspace(os.getenv("AGENT_WORKSPACE"))

    database = forge.sdk.db.AgentDB(database_name, debug_enabled=False)

    # Pass the appropriate router here if not using the default one.
    agent = Agent(database=database, workspace=workspace)
    return agent.get_agent_app()

app = get_app()
