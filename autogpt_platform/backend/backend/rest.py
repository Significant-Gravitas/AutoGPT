from backend.api.rest_api import AgentServer
from backend.app import run_processes


def main():
    """
    Run all the processes required for the AutoGPT-server REST API.
    """
    run_processes(AgentServer())


if __name__ == "__main__":
    main()
