from backend.app import run_processes
from backend.executor import DatabaseManager, ExecutionScheduler
from backend.server.rest_api import AgentServer


def main():
    """
    Run all the processes required for the AutoGPT-server REST API.
    """
    run_processes(
        DatabaseManager(),
        ExecutionScheduler(),
        AgentServer(),
    )


if __name__ == "__main__":
    main()
