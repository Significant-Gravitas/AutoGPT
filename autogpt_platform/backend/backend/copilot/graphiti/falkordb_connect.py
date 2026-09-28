"""Building FalkorDB clients off the event loop, with transport deadlines.

falkordb's asyncio client probes the server for cluster mode with a
synchronous INFO in its constructor, so a client built on the event loop
stalls every coroutine there until the server answers. Every client the
memory code uses is built on a small pool of its own (``build_off_loop``):
the cached Graphiti clients (``client.py``) are built whole on it, and a
driver made without a client (``falkordb_driver.AutoGPTFalkorDriver``, as
``open_driver`` makes) gets a ``DeferredFalkorDB``, which builds on it at
the first command. ``new_falkordb_client`` gives every client the
transport deadlines.
"""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from falkordb.asyncio import FalkorDB

from .config import graphiti_config

_T = TypeVar("_T")

# Where every FalkorDB client is built (``build_off_loop``). Building one
# sends a synchronous INFO (falkordb's cluster probe), so an unresponsive
# server holds the building thread until the transport deadlines end it; a
# small pool of its own bounds how many threads that can hold and keeps them
# off the loop's default executor, which DNS lookups and ``asyncio.to_thread``
# share.
_CONNECT_EXECUTOR = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="falkordb-connect"
)


class DeferredFalkorDB(FalkorDB):
    """A FalkorDB client that is built off the event loop, on first use.

    falkordb's ``FalkorDB.__init__`` probes the server for cluster mode with
    a synchronous INFO, so one built on the event loop stalls every
    coroutine there until the server answers. This one does no I/O when it
    is made. The first command builds the real client (``build``) on the
    connect pool (``build_off_loop``); concurrent first commands share that
    build, and a caller that stops waiting leaves it running for the next.
    The build is settled when it finishes, whether or not anyone still waits
    for it (``_settle``): a client it built is kept, and a build that failed
    is dropped, its error read, so the next command builds afresh. After
    that, every command goes through the real client.

    Deferred are the entry points the drivers reach: ``execute_command``,
    which every graph selected from this client sends its commands through
    (``select_graph`` is falkordb's own), ``list_graphs`` and ``aclose``.
    """

    def __init__(self, build: Callable[[], FalkorDB]) -> None:
        # No ``super().__init__()``: that constructor is the blocking probe
        # this class exists to defer.
        self._build = build
        self._client: FalkorDB | None = None
        self._building: asyncio.Task[FalkorDB] | None = None

    async def connect(self) -> FalkorDB:
        """The real client, built on the connect pool the first time."""
        if self._client is not None:
            return self._client
        building = self._building
        if building is None:
            building = asyncio.get_running_loop().create_task(
                build_off_loop(self._build), name="falkordb-connect"
            )
            building.add_done_callback(self._settle)
            self._building = building
        # Shielded: a caller that stops waiting leaves the build running.
        return await asyncio.shield(building)

    def _settle(self, building: "asyncio.Task[FalkorDB]") -> None:
        """Called when a build finishes, by the loop rather than by a
        waiter, since every waiter may have given up: keep the client it
        built, or drop the build that failed or was cancelled. Reading the
        error here means asyncio never reports it as unretrieved."""
        failed = building.cancelled() or building.exception() is not None
        if self._building is not building:
            return
        self._building = None
        if not failed:
            self._client = building.result()

    async def execute_command(self, *args: Any, **options: Any) -> Any:
        client = await self.connect()
        return await client.execute_command(*args, **options)

    async def list_graphs(self) -> list[str]:
        client = await self.connect()
        return await client.list_graphs()

    async def aclose(self) -> None:
        """Close the real client, if one was built. A build still running
        has opened no connection that outlives it (redis connects on the
        first command), so there is nothing else to release."""
        if self._client is not None:
            await self._client.aclose()


async def build_off_loop(build: Callable[[], _T]) -> _T:
    """Run a blocking client build on ``_CONNECT_EXECUTOR`` and await it.

    For construction that connects to FalkorDB (``new_falkordb_client``) or
    is otherwise too slow for the event loop. Cancelling the await does not
    stop the thread; the transport deadlines end any network wait in it.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_CONNECT_EXECUTOR, build)


def new_falkordb_client(
    host: str | None = None,
    port: int | None = None,
    *,
    username: str | None = None,
    password: str | None = None,
) -> FalkorDB:
    """A FalkorDB client with the transport deadlines. Blocking.

    falkordb's constructor probes the server for cluster mode with a
    synchronous INFO, so this does network I/O on the calling thread: call
    it off the event loop (``build_off_loop``, ``DeferredFalkorDB``).
    ``falkordb_socket_connect_timeout`` bounds opening each connection and
    ``falkordb_socket_timeout`` each reply, for every command on the client,
    the probe included: a command whose reply comes later fails with a
    timeout, and a write that fails that way may already have committed.
    They also end a thread whose await was given up and a connection to a
    server that stopped answering. The chat's asyncio budgets are separate
    and shorter. ``host``, ``port`` and ``password`` default to graphiti
    config.
    """
    return FalkorDB(
        host=graphiti_config.falkordb_host if host is None else host,
        port=graphiti_config.falkordb_port if port is None else port,
        username=username,
        password=(
            (graphiti_config.falkordb_password or None)
            if password is None
            else password
        ),
        socket_connect_timeout=graphiti_config.falkordb_socket_connect_timeout,
        socket_timeout=graphiti_config.falkordb_socket_timeout,
    )
