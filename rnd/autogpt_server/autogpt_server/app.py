from autogpt_server.executor import ExecutionManager, ExecutionScheduler
from autogpt_server.server import AgentServer
from autogpt_server.util.process import AppProcess
from autogpt_server.util.service import PyroNameServer


def run_processes(processes: list[AppProcess], **kwargs):
    """
    Execute all processes in the app. The last process is run in the foreground.
    """
    try:
        for process in processes[:-1]:
            process.start(background=True, **kwargs)
        processes[-1].start(background=False, **kwargs)
    except Exception as e:
        for process in processes:
            process.stop()
        raise e


def main(**kwargs):
    run_processes(
        [
            PyroNameServer(),
            ExecutionScheduler(),
            ExecutionManager(pool_size=5),
            AgentServer(),
        ],
        **kwargs
    )


if __name__ == "__main__":
    main()
