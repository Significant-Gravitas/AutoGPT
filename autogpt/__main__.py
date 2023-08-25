import os

from dotenv import load_dotenv

load_dotenv()
import autogpt.forge_log

ENABLE_TRACING = os.environ.get("ENABLE_TRACING", "false").lower() == "true"

autogpt.forge_log.setup_logger()


LOG = autogpt.forge_log.CustomLogger(__name__)

if __name__ == "__main__":
    """Runs the agent server"""

    # modules are imported here so that logging is setup first
    import autogpt.agent
    import autogpt.db
    from autogpt.benchmark_integration import add_benchmark_routes
    from autogpt.workspace import LocalWorkspace

    router = add_benchmark_routes()

    database_name = os.getenv("DATABASE_STRING")
    workspace = LocalWorkspace(os.getenv("AGENT_WORKSPACE"))
    port = os.getenv("PORT")

    database = autogpt.db.AgentDB(database_name, debug_enabled=False)
    agent = autogpt.agent.Agent(database=database, workspace=workspace)

    agent.start(port=port, router=router)
