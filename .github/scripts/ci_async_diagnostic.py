import asyncio
import gc
import os
import signal
import sys
import time

import pytest

_started = 0.0
_nodeid = ""
_phase = ""


def dump_tasks():
    print(f"ASYNC DIAGNOSTIC {_nodeid} phase={_phase}", file=sys.stderr, flush=True)
    for task in [obj for obj in gc.get_objects() if isinstance(obj, asyncio.Task)]:
        if task.done():
            continue
        loop = task.get_loop()
        print(
            f"TASK {task!r} loop={id(loop)} running={loop.is_running()}",
            file=sys.stderr,
            flush=True,
        )
        awaitable = task.get_coro()
        seen = set()
        while awaitable is not None and id(awaitable) not in seen:
            seen.add(id(awaitable))
            frame = getattr(awaitable, "cr_frame", None)
            if frame is None:
                frame = getattr(awaitable, "gi_frame", None)
            if frame is not None:
                print(
                    f"  {frame.f_code.co_filename}:{frame.f_lineno}"
                    f" in {frame.f_code.co_name}",
                    file=sys.stderr,
                    flush=True,
                )
            awaitable = getattr(awaitable, "cr_await", None) or getattr(
                awaitable, "gi_yieldfrom", None
            )


def _alarm(_signum, _frame):
    dump_tasks()
    if time.monotonic() - _started >= 150:
        print(
            "DIAGNOSTIC LIMIT: stopping after 150 seconds", file=sys.stderr, flush=True
        )
        os._exit(124)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    global _started, _nodeid
    _started = time.monotonic()
    _nodeid = item.nodeid
    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, 30, 30)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    global _phase
    _phase = "setup"


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    global _phase
    _phase = "call"


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    global _phase
    _phase = "teardown"
