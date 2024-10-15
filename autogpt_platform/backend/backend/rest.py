from backend.app import run_processes
from backend.executor import ExecutionScheduler
from backend.server.db_api import DatabaseAPI
from backend.server.rest_api import AgentServer


def main():
    """
    Run all the processes required for the AutoGPT-server REST API.
    """
    run_processes(
        DatabaseAPI(),
        ExecutionScheduler(),
        AgentServer(),
    )


if __name__ == "__main__":
    main()
