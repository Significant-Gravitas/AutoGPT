from multiprocessing import freeze_support, set_start_method

import Pyro5.api as pyro
from tenacity import retry, stop_after_attempt, wait_exponential

from autogpt_server.executor import ExecutionManager, ExecutionScheduler
from autogpt_server.server import AgentServer
from autogpt_server.util.process import AppProcess
from autogpt_server.util.service import PyroNameServer


def get_config_and_secrets():
    from autogpt_server.util.settings import Settings

    settings = Settings()
    return settings


@retry(stop=stop_after_attempt(30), wait=wait_exponential(multiplier=1, min=1, max=30))
def wait_for_nameserver():
    pyro.locate_ns(host="localhost", port=9090)
    print("NameServer is ready")


def run_processes(processes: list[AppProcess], **kwargs):
    """
    Execute all processes in the app. The last process is run in the foreground.
    """
    try:
        # Start NameServer first
        processes[0].start(background=True, **kwargs)

        # Wait for NameServer to be ready
        wait_for_nameserver()

        # Start other processes
        for process in processes[1:-1]:
            process.start(background=True, **kwargs)

        # Run the last process in the foreground
        processes[-1].start(background=False, **kwargs)
    except Exception as e:
        for process in processes:
            process.stop()
        raise e


def main(**kwargs):
    set_start_method("spawn", force=True)
    freeze_support()

    run_processes(
        [
            PyroNameServer(),
            ExecutionManager(),
            ExecutionScheduler(),
            AgentServer(),
        ],
        **kwargs
    )


if __name__ == "__main__":
    main()
