from backend.app import run_processes
from backend.scheduler_service import scheduler_service


def main():
    """
    Run all the processes required for the AutoGPT-server Scheduling System.
    """
    run_processes(
        scheduler_service(),
    )


if __name__ == "__main__":
    main()
