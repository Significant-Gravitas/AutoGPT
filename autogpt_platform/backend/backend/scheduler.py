from backend.app import run_processes
from backend.executor.scheduler import Scheduler
from backend.util.logging import configure_logging


def main():
    """
    Run all the processes required for the AutoGPT-server Scheduling System.
    """
    configure_logging()
    run_processes(Scheduler())


if __name__ == "__main__":
    main()
