import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.util.process import AppProcess

logger = logging.getLogger(__name__)


def run_processes(*processes: "AppProcess", **kwargs):
    """
    Execute all processes in the app. The last process is run in the foreground.
    Includes enhanced error handling and process lifecycle management.
    """
    try:
        # Run all processes except the last one in the background.
        for process in processes[:-1]:
            process.start(background=True, **kwargs)

        # Run the last process in the foreground.
        processes[-1].start(background=False, **kwargs)
    finally:
        for process in processes:
            try:
                process.stop()
            except Exception as e:
                logger.exception(f"[{process.service_name}] unable to stop: {e}")


def main(**kwargs):
    """
    Run all the processes required for the AutoGPT-server (REST and WebSocket APIs).
    """

    from backend.executor import DatabaseManager, ExecutionManager, Scheduler
    from backend.notifications import NotificationManager
    from backend.server.rest_api import AgentServer
    from backend.server.ws_api import WebsocketServer

    run_processes(
        DatabaseManager(),
        ExecutionManager(),
        Scheduler(),
        NotificationManager(),
        WebsocketServer(),
        AgentServer(),
        **kwargs,
    )


if __name__ == "__main__":
    main()
