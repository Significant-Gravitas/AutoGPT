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
    active_processes = []
    try:
        # Start background processes with health checks
        for process in processes[:-1]:
            try:
                process.start(background=True, **kwargs)
                active_processes.append(process)
                process.health_check()
            except Exception as e:
                logger.error(
                    f"Failed to start process {process.service_name}: {str(e)}"
                )
                # Cleanup already started processes
                for p in active_processes:
                    try:
                        p.stop()
                    except Exception as cleanup_error:
                        logger.error(
                            f"Error during cleanup of {p.service_name}: {str(cleanup_error)}"
                        )
                raise RuntimeError(f"Process startup failed: {str(e)}") from e

        # Run the last process in the foreground
        try:
            processes[-1].start(background=False, **kwargs)
        except Exception as e:
            logger.error(f"Foreground process failed: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Process execution failed: {str(e)}")
        raise
    finally:
        # Ensure all processes are properly stopped
        for process in active_processes:
            try:
                process.stop()
                logger.info(f"Successfully stopped process {process.service_name}")
            except Exception as e:
                logger.error(f"Failed to stop process {process.service_name}: {str(e)}")


def main(**kwargs):
    """
    Run all the processes required for the AutoGPT-server (REST and WebSocket APIs).
    """

    from backend.executor import DatabaseManager, ExecutionManager, ExecutionScheduler
    from backend.notifications import NotificationManager
    from backend.server.rest_api import AgentServer
    from backend.server.ws_api import WebsocketServer

    run_processes(
        DatabaseManager(),
        ExecutionManager(),
        ExecutionScheduler(),
        NotificationManager(),
        WebsocketServer(),
        AgentServer(),
        **kwargs,
    )


if __name__ == "__main__":
    main()
