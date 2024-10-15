from backend.app import run_processes
from backend.executor import ExecutionManager
from backend.server.db_api import DatabaseAPI


def main():
    """
    Run all the processes required for the AutoGPT-server REST API.
    """
    run_processes(
        DatabaseAPI(),
        ExecutionManager(),
    )


if __name__ == "__main__":
    main()
