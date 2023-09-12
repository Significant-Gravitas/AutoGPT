import os

from dotenv import load_dotenv

load_dotenv()
import forge.sdk.forge_log

forge.sdk.forge_log.setup_logger()


LOG = forge.sdk.forge_log.ForgeLogger(__name__)

if __name__ == "__main__":
    """Runs the agent server"""

    # modules are imported here so that logging is setup first
    import forge.agent
    import forge.sdk.db
    from forge.sdk.workspace import LocalWorkspace

    database_name = os.getenv("DATABASE_STRING")
    workspace = LocalWorkspace(os.getenv("AGENT_WORKSPACE"))
    port = os.getenv("PORT")

    database = forge.sdk.db.AgentDB(database_name, debug_enabled=True)
    agent = forge.agent.forgeAgent(database=database, workspace=workspace)

    agent.start(port=port)
