import uvicorn
import backend.util.settings

def main():
    """
    Run all the processes required for the AutoGPT-server REST API.
    """
    uvicorn.run(
        "backend.server.app:app",
        reload=True,
        host=backend.util.settings.Config().agent_api_host,
        port=backend.util.settings.Config().agent_api_port,
    )

if __name__ == "__main__":
    main()
