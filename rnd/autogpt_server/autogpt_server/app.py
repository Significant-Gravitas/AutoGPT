from multiprocessing import freeze_support, set_start_method
from typing import TYPE_CHECKING

import Pyro5.api as pyro
from tenacity import retry, stop_after_attempt, wait_exponential

from .util.logging import configure_logging

if TYPE_CHECKING:
    from autogpt_server.util.process import AppProcess

def run_processes(processes: list["AppProcess"], **kwargs):
    """
    Execute all processes in the app. The last process is run in the foreground.
    """
    try:
        processes[0].start(background=False, **kwargs)
    except Exception as e:
        for process in processes:
            process.stop()
        raise e


def main(**kwargs):
    set_start_method("spawn", force=True)
    freeze_support()
    configure_logging()

    from autogpt_server.executor import ExecutionScheduler
    from autogpt_server.server import AgentServer
    from autogpt_server.util.service import PyroNameServer

    run_processes(
        [
            AgentServer(),
        ],
        **kwargs
    )

def execution_manager(**kwargs):
    set_start_method("spawn", force=True)
    freeze_support()
    configure_logging()

    from autogpt_server.executor import ExecutionManager

    run_processes(
        [
            ExecutionManager(),
        ],
        **kwargs
    )

if __name__ == "__main__":
    main()
