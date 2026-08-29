from backend.app import run_processes
from backend.notifications.notifications import NotificationManager


def main():
    """
    Run the AutoGPT-server Notification Service.
    """
    run_processes(
        NotificationManager(),
    )


if __name__ == "__main__":
    main()
