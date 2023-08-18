import os

from dotenv import load_dotenv

import autogpt.agent
import autogpt.db
from autogpt.benchmark_integration import add_benchmark_routes
from autogpt.workspace import LocalWorkspace

if __name__ == "__main__":
    """Runs the agent server"""
    load_dotenv()
    router = add_benchmark_routes()

    database_name = os.getenv("DATABASE_STRING")
    workspace = LocalWorkspace(os.getenv("AGENT_WORKSPACE"))
    print(database_name)
    port = os.getenv("PORT")

    database = autogpt.db.AgentDB(database_name)
    agent = autogpt.agent.Agent(database=database, workspace=workspace)

    agent.start(port=port, router=router)
