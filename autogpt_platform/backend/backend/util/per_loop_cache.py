"""Per-event-loop caching for clients bound to the loop that first uses them.

An ``AsyncOpenAI`` / ``httpx.AsyncClient`` keeps idle keep-alive connections
whose streams are bound to the event loop that opened them. Shared across
loops (the copilot and graph executors run one loop per worker thread), a
call on loop B can pick up a connection opened on loop A: the request is
written to the socket, reading the reply raises "... is bound to a different
event loop", and the SDK retries the call on a new connection, so the
provider receives (and bills) the call twice while the caller sees one
success. ``per_loop_cached`` gives every running loop its own instance.
"""

import asyncio
import logging
import threading
from collections.abc import AsyncGenerator, Callable, Hashable
from functools import update_wrapper
from typing import Generic, ParamSpec, Protocol, TypeVar
from weakref import WeakKeyDictionary

logger = logging.getLogger(__name__)

P = ParamSpec("P")


class AsyncClosable(Protocol):
    async def close(self) -> None: ...


C = TypeVar("C", bound=AsyncClosable | None)


def per_loop_cached(factory: Callable[P, C]) -> "PerLoopCached[P, C]":
    """Cache ``factory``'s result per running event loop.

    Each loop gets its own instance, closed on that loop when the loop shuts
    down its async generators (``asyncio.run`` and ``asyncio.Runner`` do), so
    a loop created per job doesn't leave a client behind. A loop that is only
    stopped keeps its instance for the life of the process. Called with no
    running loop, the factory runs uncached.
    """
    return PerLoopCached(factory)


class PerLoopCached(Generic[P, C]):
    def __init__(self, factory: Callable[P, C]) -> None:
        self._factory = factory
        self._lock = threading.Lock()
        # Keyed on the loop itself, not id(loop): a dead loop's id can be
        # reused by a new loop, which would inherit the dead loop's client.
        self._values: WeakKeyDictionary[
            asyncio.AbstractEventLoop, dict[Hashable, C]
        ] = WeakKeyDictionary()
        self._finalizers: WeakKeyDictionary[
            asyncio.AbstractEventLoop, AsyncGenerator[None, None]
        ] = WeakKeyDictionary()
        update_wrapper(self, factory)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> C:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return self._factory(*args, **kwargs)
        key = (args, tuple(sorted(kwargs.items())))
        with self._lock:
            values = self._values.get(loop)
            if values is None:
                values = self._values[loop] = {}
                finalizer = self._close_on_loop_shutdown(values)
                _start_loop_finalizer(finalizer)
                self._finalizers[loop] = finalizer
        # Only this loop's thread reaches ``values`` (a running loop belongs
        # to one thread), and there is no await between check and store.
        if key not in values:
            values[key] = self._factory(*args, **kwargs)
        return values[key]

    def cache_clear(self) -> None:
        """Forget every loop's instance; the next call on each loop builds a new one."""
        with self._lock:
            caches = list(self._values.values())
        for values in caches:
            values.clear()

    async def _close_on_loop_shutdown(
        self, values: dict[Hashable, C]
    ) -> AsyncGenerator[None, None]:
        try:
            yield
        finally:
            loop = asyncio.get_running_loop()
            with self._lock:
                if self._values.get(loop) is values:
                    del self._values[loop]
                    self._finalizers.pop(loop, None)
            clients = [client for client in values.values() if client is not None]
            values.clear()
            for client in clients:
                try:
                    await client.close()
                except Exception:
                    logger.warning(
                        "Failed to close a per-loop client at loop shutdown",
                        exc_info=True,
                    )


def _start_loop_finalizer(finalizer: AsyncGenerator[None, None]) -> None:
    """Advance ``finalizer`` to its ``yield`` from synchronous code.

    The first step registers the generator with the running loop, whose
    ``shutdown_asyncgens()`` then closes it on that loop, running its
    ``finally``. Nothing before the ``yield`` awaits, so one ``send`` completes
    the step (an async generator's ``yield`` ends it with ``StopIteration``).
    """
    try:
        finalizer.asend(None).send(None)
    except StopIteration:
        return
    raise RuntimeError("a loop finalizer must not await before its first yield")
