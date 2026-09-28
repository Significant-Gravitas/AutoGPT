"""Tests for building FalkorDB clients off the event loop
(``falkordb_connect.py``)."""

import asyncio
import gc
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import falkordb_connect


def test_every_client_carries_the_transport_deadlines(monkeypatch) -> None:
    """falkordb defaults both to None, and its constructor probes the server
    with a synchronous INFO: without deadlines an unresponsive server holds
    an abandoned worker thread indefinitely."""
    monkeypatch.setattr(
        falkordb_connect.graphiti_config, "falkordb_socket_connect_timeout", 0.7
    )
    monkeypatch.setattr(
        falkordb_connect.graphiti_config, "falkordb_socket_timeout", 1.3
    )
    with patch.object(falkordb_connect, "FalkorDB") as falkordb:
        assert falkordb_connect.new_falkordb_client() is falkordb.return_value
        falkordb_connect.new_falkordb_client("falkordb.internal", 7000)

    configured, explicit = (call.kwargs for call in falkordb.call_args_list)
    for kwargs in (configured, explicit):
        assert kwargs["socket_connect_timeout"] == 0.7
        assert kwargs["socket_timeout"] == 1.3
    assert configured["host"] == falkordb_connect.graphiti_config.falkordb_host
    assert configured["port"] == falkordb_connect.graphiti_config.falkordb_port
    assert (explicit["host"], explicit["port"]) == ("falkordb.internal", 7000)


class TestDeferredFalkorDB:
    @pytest.mark.asyncio
    async def test_concurrent_first_commands_share_one_build(self) -> None:
        builds: list[int] = []
        real = MagicMock()
        real.execute_command = AsyncMock(return_value="PONG")

        def build() -> MagicMock:
            builds.append(1)
            time.sleep(0.1)
            return real

        client = falkordb_connect.DeferredFalkorDB(build)
        results = await asyncio.gather(
            *(client.execute_command("PING") for _ in range(5))
        )

        assert results == ["PONG"] * 5
        assert len(builds) == 1

    @pytest.mark.asyncio
    async def test_a_failed_build_is_retried_by_the_next_command(self) -> None:
        attempts: list[int] = []
        real = MagicMock()
        real.list_graphs = AsyncMock(return_value=["user_a"])

        def build() -> MagicMock:
            attempts.append(1)
            if len(attempts) == 1:
                raise ConnectionError("connection refused")
            return real

        client = falkordb_connect.DeferredFalkorDB(build)
        with pytest.raises(ConnectionError):
            await client.list_graphs()

        assert await client.list_graphs() == ["user_a"]
        assert len(attempts) == 2

    @pytest.mark.asyncio
    async def test_a_caller_that_gives_up_leaves_the_build_to_the_next(
        self,
    ) -> None:
        release = threading.Event()
        builds: list[int] = []
        real = MagicMock()
        real.execute_command = AsyncMock(return_value=1)

        def build() -> MagicMock:
            builds.append(1)
            release.wait(5)
            return real

        client = falkordb_connect.DeferredFalkorDB(build)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(client.execute_command("PING"), timeout=0.05)
        release.set()

        assert await client.execute_command("PING") == 1
        assert len(builds) == 1

    @staticmethod
    def _abandoned_build(
        release: threading.Event, real: MagicMock | None = None
    ) -> tuple[Any, list[int]]:
        """A build whose first attempt waits for ``release`` and then fails;
        any later attempt returns ``real``."""
        attempts: list[int] = []

        def build() -> MagicMock:
            attempts.append(1)
            if len(attempts) == 1:
                release.wait(5)
                raise ConnectionError("the server stopped answering")
            assert real is not None
            return real

        return build, attempts

    @pytest.mark.asyncio
    async def test_a_build_that_fails_after_its_caller_gave_up_is_read(
        self,
    ) -> None:
        """The caller gives up, the build then fails with nobody waiting for
        it, and the driver is closed and dropped, as a cancelled request's
        ``finally`` does. The failure was read when the build settled, so the
        loop's exception handler hears nothing ("Task exception was never
        retrieved")."""
        loop = asyncio.get_running_loop()
        reported: list[dict[str, Any]] = []
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: reported.append(context))
        release = threading.Event()
        build, _ = self._abandoned_build(release)
        try:
            client = falkordb_connect.DeferredFalkorDB(build)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(client.execute_command("PING"), timeout=0.05)
            building = client._building
            assert building is not None
            await client.aclose()
            release.set()
            await asyncio.wait({building}, timeout=5)
            assert building.done() and client._building is None
            del client, building
            gc.collect()
            await asyncio.sleep(0.01)
        finally:
            loop.set_exception_handler(previous)

        assert reported == []

    @pytest.mark.asyncio
    async def test_the_first_command_after_an_abandoned_failed_build_succeeds(
        self,
    ) -> None:
        """The caller gives up, the build then fails with nobody waiting, and
        the server comes back: the very next command builds afresh, instead
        of raising the dead build's error once more."""
        release = threading.Event()
        real = MagicMock()
        real.execute_command = AsyncMock(return_value="PONG")
        build, attempts = self._abandoned_build(release, real)
        client = falkordb_connect.DeferredFalkorDB(build)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(client.execute_command("PING"), timeout=0.05)
        building = client._building
        assert building is not None
        release.set()
        await asyncio.wait({building}, timeout=5)

        assert await client.execute_command("PING") == "PONG"
        assert len(attempts) == 2

    @pytest.mark.asyncio
    async def test_a_cancelled_build_is_dropped(self) -> None:
        """A build cancelled outright (not a caller giving up, which the
        shield absorbs) is dropped too, so the next command builds afresh
        instead of raising ``CancelledError``."""
        release = threading.Event()
        real = MagicMock()
        real.execute_command = AsyncMock(return_value="PONG")
        build, attempts = self._abandoned_build(release, real)
        client = falkordb_connect.DeferredFalkorDB(build)
        waiter = asyncio.create_task(client.execute_command("PING"))
        await asyncio.sleep(0.05)
        building = client._building
        assert building is not None
        building.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await asyncio.wait({building}, timeout=5)
        release.set()

        assert await client.execute_command("PING") == "PONG"
        assert len(attempts) == 2

    @pytest.mark.asyncio
    async def test_close_releases_the_built_client_and_builds_none(self) -> None:
        unused = falkordb_connect.DeferredFalkorDB(
            MagicMock(side_effect=AssertionError("built"))
        )
        await unused.aclose()

        real = MagicMock()
        real.execute_command = AsyncMock()
        real.aclose = AsyncMock()
        used = falkordb_connect.DeferredFalkorDB(lambda: real)
        await used.execute_command("PING")
        await used.aclose()

        real.aclose.assert_awaited_once()
