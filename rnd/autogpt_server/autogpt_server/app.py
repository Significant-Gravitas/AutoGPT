from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogpt_server.util.process import AppProcess


def run_processes(*processes: "AppProcess", **kwargs):
    """
    Execute all processes in the app. The last process is run in the foreground.
    """
    try:
        for process in processes[:-1]:
            process.start(background=True, **kwargs)

        # Run the last process in the foreground
        processes[-1].start(background=False, **kwargs)
    finally:
        for process in processes:
            process.stop()


def main(**kwargs):
    """
    Run all the processes required for the AutoGPT-server (REST and WebSocket APIs).
    """

    from autogpt_server.executor import ExecutionManager, ExecutionScheduler
    from autogpt_server.server import AgentServer, WebsocketServer

    run_processes(
        ExecutionManager(),
        ExecutionScheduler(),
        WebsocketServer(),
        AgentServer(),
        **kwargs,
    )


if __name__ == "__main__":
    main()
