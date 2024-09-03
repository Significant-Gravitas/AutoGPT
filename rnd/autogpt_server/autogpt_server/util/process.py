import os
import signal
import sys
from abc import ABC, abstractmethod
from multiprocessing import Process, set_start_method
from typing import Optional


class AppProcess(ABC):
    """
    A class to represent an object that can be executed in a background process.
    """

    process: Optional[Process] = None
    set_start_method("spawn", force=True)

    # Methods that are executed INSIDE the process #

    @abstractmethod
    def run(self):
        """
        The method that will be executed in the process.
        """
        pass

    def cleanup(self):
        """
        Implement this method on a subclass to do post-execution cleanup,
        e.g. disconnecting from a database or terminating child processes.
        """
        pass

    def execute_run_command(self, silent):
        signal.signal(signal.SIGTERM, self._self_terminate)

        try:
            if silent:
                sys.stdout = open(os.devnull, "w")
                sys.stderr = open(os.devnull, "w")
            else:
                from .logging import configure_logging

                configure_logging()
            self.run()
        except KeyboardInterrupt or SystemExit as e:
            print(f"Process terminated: {e}")

    def _self_terminate(self, signum: int, frame):
        self.cleanup()
        sys.exit(0)

    # Methods that are executed OUTSIDE the process #

    def __enter__(self):
        self.start(background=True)
        return self

    def __exit__(self, *args, **kwargs):
        self.stop()

    def start(self, background: bool = False, silent: bool = False, **proc_args) -> int:
        """
        Start the background process.
        Args:
            background: Whether to run the process in the background.
            silent: Whether to disable stdout and stderr.
            proc_args: Additional arguments to pass to the process.
        Returns:
            the process id or 0 if the process is not running in the background.
        """
        if not background:
            self.execute_run_command(silent)
            return 0

        self.process = Process(
            name=self.__class__.__name__,
            target=self.execute_run_command,
            args=(silent,),
            **proc_args,
        )
        self.process.start()
        return self.process.pid or 0

    def stop(self):
        """
        Stop the background process.
        """
        if not self.process:
            return

        self.process.terminate()
        self.process.join()
        self.process = None
