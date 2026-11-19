import gc
import time
from contextlib import contextmanager


@contextmanager
def observe_startup(disable: bool):
    enabled = gc.isenabled()
    started = time.perf_counter()
    collections = []
    active = {}
    result = {"disabled_during_startup": disable, "initially_enabled": enabled}

    def record(phase, info):
        generation = info["generation"]
        if phase == "start":
            active[generation] = time.perf_counter()
        elif generation in active:
            collections.append(
                {
                    **info,
                    "seconds": time.perf_counter() - active.pop(generation),
                }
            )

    gc.callbacks.append(record)
    if disable:
        gc.disable()
    try:
        yield result
    finally:
        if enabled:
            gc.enable()
        else:
            gc.disable()
        gc.callbacks.remove(record)
        result.update(
            seconds=time.perf_counter() - started,
            finally_enabled=gc.isenabled(),
            collections=collections,
            gc_seconds=sum(item["seconds"] for item in collections),
        )
