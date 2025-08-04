from backend.app import run_processes
from backend.executor.scheduler import Scheduler
from backend.notifications.notifications import NotificationManager


def main():
    """
    Run all the processes required for the AutoGPT-server Scheduling System.
    """
    run_processes(
        NotificationManager(),
        Scheduler(),
    )


if __name__ == "__main__":
    main()
