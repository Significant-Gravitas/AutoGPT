from backend.app import run_processes
from backend.executor import ExecutionManager
from backend.util.logging import configure_logging


def main():
    """
    Run all the processes required for the AutoGPT-server REST API.
    """
    configure_logging()
    run_processes(ExecutionManager())


if __name__ == "__main__":
    main()
