from backend.app import run_processes
from backend.executor import ExecutionManager


def main():
    """
    Run all the processes required for the AutoGPT-server REST API.
    """
    run_processes(
        ExecutionManager(),
    )


if __name__ == "__main__":
    main()
