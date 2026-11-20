"""Find async Redis Cluster clients a test left open on an event loop that
has closed.

redis-py's asyncio ``RedisCluster`` and its ``ClusterNode`` report themselves
when garbage-collected while still open ("Unclosed RedisCluster client",
"Unclosed ClusterNode object"), to the exception handler of whatever event
loop is running then. A client left open on a loop that has closed can no
longer be closed, so its report lands in a later, unrelated test, and fails
any test that collects garbage while watching its loop's exception handler.
"""

import asyncio
import os
import traceback
import weakref
from collections.abc import Awaitable, Callable
from typing import Any

from redis.asyncio.cluster import RedisCluster

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_PACKAGE = os.path.join(_BACKEND_ROOT, "backend") + os.sep

ExecuteCommand = Callable[..., Awaitable[Any]]


class RedisClientGuard:
    """Remembers each client a test sends a command through and the loop it
    sent on, then names the ones left open there once that loop has closed.

    Only clients ``enforced`` accepts are named for failing the test; the rest
    left open are kept in ``tolerated`` after each ``stop()``, to report."""

    def __init__(
        self, enforced: Callable[[RedisCluster], bool] = lambda client: True
    ) -> None:
        # The client strongly, so one the test dropped is still here to check
        # however the garbage collector's timing falls; its loop weakly, so
        # watching keeps no loop alive. A loop already collected has closed.
        self._used: dict[
            int, tuple[RedisCluster, weakref.ref[asyncio.AbstractEventLoop], str]
        ] = {}
        self._watching = False
        self._enforced = enforced
        self.tolerated: list[str] = []

    def start(self) -> None:
        self._used.clear()
        self._watching = True

    def stop(self) -> list[str]:
        """Each enforced client left open on a loop that has closed. The others
        left open go to ``tolerated``."""
        self._watching = False
        used, self._used = self._used, {}
        left_open = [
            (client, described)
            for client, loop, described in used.values()
            if _closed(loop()) and is_open(client)
        ]
        self.tolerated = [
            described for client, described in left_open if not self._enforced(client)
        ]
        return [described for client, described in left_open if self._enforced(client)]

    def wrap(self, execute_command: ExecuteCommand) -> ExecuteCommand:
        """``execute_command``, recording each client on its first command of
        the test."""

        async def recorded_execute_command(
            client: RedisCluster, *args: Any, **kwargs: Any
        ) -> Any:
            if self._watching and id(client) not in self._used:
                self._used[id(client)] = (
                    client,
                    weakref.ref(asyncio.get_running_loop()),
                    f"Redis Cluster client first used at {_caller()}",
                )
            return await execute_command(client, *args, **kwargs)

        return recorded_execute_command


def _closed(loop: asyncio.AbstractEventLoop | None) -> bool:
    return loop is None or loop.is_closed()


def is_open(client: RedisCluster) -> bool:
    """Whether redis-py would report ``client`` when collecting it: set up and
    never closed, or with a node still connected."""
    nodes = [
        *client.nodes_manager.nodes_cache.values(),
        *client.nodes_manager.startup_nodes.values(),
    ]
    return not client._initialize or any(
        connection.is_connected for node in nodes for connection in node._connections
    )


def left_open_report(nodeid: str, left_open: list[str]) -> str:
    """What a test that left clients open is told, and how to fix it."""
    return (
        f"{nodeid} left {len(left_open)} async Redis Cluster client(s) open "
        "on an event loop that has closed:\n"
        + "\n".join(f"  {client}" for client in left_open)
        + "\nTheir connections can no longer be closed. Once such a client is "
        'garbage-collected, it reports "Unclosed ClusterNode object" to the loop '
        "of whatever test is running, which fails any test watching its loop's "
        "exception handler. Close each client before its loop ends: `aclose()` "
        "it, or let `asyncio.run` / `loop.shutdown_asyncgens()` close a client "
        "made through `backend.data.redis_client`."
    )


def _caller() -> str:
    """The innermost backend frame outside this module."""
    for frame, line in traceback.walk_stack(None):
        path = frame.f_code.co_filename
        if path.startswith(_PACKAGE) and path != __file__:
            return f"{os.path.relpath(path, _BACKEND_ROOT)}:{line} in {frame.f_code.co_name}"
    return "a caller outside the backend"
