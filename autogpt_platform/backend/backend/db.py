from backend.app import run_processes
from backend.executor import DatabaseManager


def main():
    """
    Run all the processes required for the AutoGPT-server REST API.
    """
    run_processes(DatabaseManager())


if __name__ == "__main__":
    main()
