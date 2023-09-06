import os

from dotenv import load_dotenv

load_dotenv()
import autogpt.sdk.forge_log

autogpt.sdk.forge_log.setup_logger()


LOG = autogpt.sdk.forge_log.ForgeLogger(__name__)

if __name__ == "__main__":
    """Runs the agent server"""

    # modules are imported here so that logging is setup first
    import autogpt.agent
    import autogpt.sdk.db
    from autogpt.sdk.workspace import LocalWorkspace

    database_name = os.getenv("DATABASE_STRING")
    workspace = LocalWorkspace(os.getenv("AGENT_WORKSPACE"))
    port = os.getenv("PORT")

    database = autogpt.sdk.db.AgentDB(database_name, debug_enabled=True)
    agent = autogpt.agent.AutoGPTAgent(database=database, workspace=workspace)

    agent.start(port=port)
