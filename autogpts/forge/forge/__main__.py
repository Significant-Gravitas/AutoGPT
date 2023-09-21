import os

from dotenv import load_dotenv

load_dotenv()
import forge.sdk.forge_log

forge.sdk.forge_log.setup_logger()


LOG = forge.sdk.forge_log.ForgeLogger(__name__)

logo = """\n\n
       d8888          888             .d8888b.  8888888b. 88888888888 
      d88888          888            d88P  Y88b 888   Y88b    888     
     d88P888          888            888    888 888    888    888     
    d88P 888 888  888 888888 .d88b.  888        888   d88P    888     
   d88P  888 888  888 888   d88""88b 888  88888 8888888P"     888     
  d88P   888 888  888 888   888  888 888    888 888           888     
 d8888888888 Y88b 888 Y88b. Y88..88P Y88b  d88P 888           888     
d88P     888  "Y88888  "Y888 "Y88P"   "Y8888P88 888           888     
                                                                      
                                                                      
                                                                      
                8888888888                                            
                888                                                   
                888                                                   
                8888888  .d88b.  888d888 .d88b.   .d88b.              
                888     d88""88b 888P"  d88P"88b d8P  Y8b             
                888     888  888 888    888  888 88888888             
                888     Y88..88P 888    Y88b 888 Y8b.                 
                888      "Y88P"  888     "Y88888  "Y8888              
                                             888                      
                                        Y8b d88P                      
                                         "Y88P"                v0.1.0
\n"""

if __name__ == "__main__":
    """Runs the agent server"""

    # modules are imported here so that logging is setup first
    import forge.agent
    import forge.sdk.db
    from forge.sdk.workspace import LocalWorkspace

    print(logo)
    database_name = os.getenv("DATABASE_STRING")
    workspace = LocalWorkspace(os.getenv("AGENT_WORKSPACE"))
    port = os.getenv("PORT", 8000)

    database = forge.sdk.db.AgentDB(database_name, debug_enabled=False)
    agent = forge.agent.ForgeAgent(database=database, workspace=workspace)

    agent.start(port=port)
