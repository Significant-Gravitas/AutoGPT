"""
IO/concurrency helpers for `tqdm.contrib`.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from ..auto import tqdm as tqdm_auto

__author__ = {"github.com/": ["casperdcl"]}
__all__ = ['MonoWorker']


class MonoWorker(object):
    """
    Supports one running task and one waiting task.
    The waiting task is the most recent submitted (others are discarded).
    """
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.futures = deque([], 2)

    def submit(self, func, *args, **kwargs):
        """`func(*args, **kwargs)` may replace currently waiting task."""
        futures = self.futures
        if len(futures) == futures.maxlen:
            running = futures.popleft()
            if not running.done():
                if len(futures):  # clear waiting
                    waiting = futures.pop()
                    waiting.cancel()
                futures.appendleft(running)  # re-insert running
        try:
            waiting = self.pool.submit(func, *args, **kwargs)
        except Exception as e:
            tqdm_auto.write(str(e))
        else:
            futures.append(waiting)
            return waiting
