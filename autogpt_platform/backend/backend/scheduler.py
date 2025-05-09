from backend.app import run_processes
from backend.executor.scheduler import Scheduler


def main():
    """
    Run all the processes required for the AutoGPT-server Scheduling System.
    """
    run_processes(Scheduler())


if __name__ == "__main__":
    main()
