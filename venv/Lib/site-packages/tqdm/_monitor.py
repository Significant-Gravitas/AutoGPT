import atexit
from threading import Event, Thread, current_thread
from time import time
from warnings import warn

__all__ = ["TMonitor", "TqdmSynchronisationWarning"]


class TqdmSynchronisationWarning(RuntimeWarning):
    """tqdm multi-thread/-process errors which may cause incorrect nesting
    but otherwise no adverse effects"""
    pass


class TMonitor(Thread):
    """
    Monitoring thread for tqdm bars.
    Monitors if tqdm bars are taking too much time to display
    and readjusts miniters automatically if necessary.

    Parameters
    ----------
    tqdm_cls  : class
        tqdm class to use (can be core tqdm or a submodule).
    sleep_interval  : float
        Time to sleep between monitoring checks.
    """
    _test = {}  # internal vars for unit testing

    def __init__(self, tqdm_cls, sleep_interval):
        Thread.__init__(self)
        self.daemon = True  # kill thread when main killed (KeyboardInterrupt)
        self.woken = 0  # last time woken up, to sync with monitor
        self.tqdm_cls = tqdm_cls
        self.sleep_interval = sleep_interval
        self._time = self._test.get("time", time)
        self.was_killed = self._test.get("Event", Event)()
        atexit.register(self.exit)
        self.start()

    def exit(self):
        self.was_killed.set()
        if self is not current_thread():
            self.join()
        return self.report()

    def get_instances(self):
        # returns a copy of started `tqdm_cls` instances
        return [i for i in self.tqdm_cls._instances.copy()
                # Avoid race by checking that the instance started
                if hasattr(i, 'start_t')]

    def run(self):
        cur_t = self._time()
        while True:
            # After processing and before sleeping, notify that we woke
            # Need to be done just before sleeping
            self.woken = cur_t
            # Sleep some time...
            self.was_killed.wait(self.sleep_interval)
            # Quit if killed
            if self.was_killed.is_set():
                return
            # Then monitor!
            # Acquire lock (to access _instances)
            with self.tqdm_cls.get_lock():
                cur_t = self._time()
                # Check tqdm instances are waiting too long to print
                instances = self.get_instances()
                for instance in instances:
                    # Check event in loop to reduce blocking time on exit
                    if self.was_killed.is_set():
                        return
                    # Only if mininterval > 1 (else iterations are just slow)
                    # and last refresh exceeded maxinterval
                    if (
                        instance.miniters > 1
                        and (cur_t - instance.last_print_t) >= instance.maxinterval
                    ):
                        # force bypassing miniters on next iteration
                        # (dynamic_miniters adjusts mininterval automatically)
                        instance.miniters = 1
                        # Refresh now! (works only for manual tqdm)
                        instance.refresh(nolock=True)
                    # Remove accidental long-lived strong reference
                    del instance
                if instances != self.get_instances():  # pragma: nocover
                    warn("Set changed size during iteration" +
                         " (see https://github.com/tqdm/tqdm/issues/481)",
                         TqdmSynchronisationWarning, stacklevel=2)
                # Remove accidental long-lived strong references
                del instances

    def report(self):
        return not self.was_killed.is_set()
