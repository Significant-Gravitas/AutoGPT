"""A database manager fake for test folders that start no database manager.

Every call fails at once with the error an unreachable database manager ends
in, instead of after the real client's retries (seconds per call at best,
close to an hour at the default retry count). What the code does with the
failure is unchanged; only the wait goes.

A conftest turns it on for its folder with::

    fake_database_manager_clients = database_manager_fixtures.fake_database_manager_clients
    database_manager = database_manager_fixtures.database_manager
"""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Iterator
from typing import Any, NoReturn

import httpx
import pytest

from backend.data.db_manager import DatabaseManagerAsyncClient, DatabaseManagerClient
from backend.util import clients, service
from backend.util.cache import clear_thread_cache

_CALLS_THIS_SESSION: Counter[str] = Counter()


class DatabaseManagerFake:
    """``calls`` lists the methods called while one test ran, including any
    from a background task an earlier test left running."""

    def __init__(self) -> None:
        self.calls: list[str] = []


_current = DatabaseManagerFake()


class _FakeDatabaseManagerClient:
    def __init__(self, client_type: type) -> None:
        self._client_type = client_type

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._client_type, name, None)
        if method is None:
            raise AttributeError(f"Method {name} not found in {self._client_type}")

        def fail() -> NoReturn:
            _current.calls.append(name)
            _CALLS_THIS_SESSION[name] += 1
            raise httpx.ConnectError(
                f"No database manager runs in these tests ({name})",
                request=httpx.Request("POST", f"http://database-manager/{name}"),
            )

        if inspect.iscoroutinefunction(method):

            async def async_method(*args: Any, **kwargs: Any) -> Any:
                fail()

            return async_method

        def sync_method(*args: Any, **kwargs: Any) -> Any:
            fail()

        return sync_method


@pytest.fixture(scope="package", autouse=True)
def fake_database_manager_clients() -> Iterator[None]:
    """Build the fake wherever a database manager client is built, so it also
    reaches modules that imported a client getter by name. A test that patches
    a getter or a ``db_accessors`` function itself still wins.

    It holds for the whole folder, not per test: a background task a test left
    running (a chat title, a cost log) reaches for the client between tests.

    A test that needs a real client, such as dream's deadline tests with
    their own transports, imports ``get_service_client`` by name, which this
    patch does not reach.

    The getters cache a client per thread, and only this thread's cache is
    cleared: a worker thread that cached a real client before the folder's
    first test keeps it, and one that cached the fake keeps that after the
    folder's last test.
    """

    real_get_service_client = service.get_service_client

    # The real getter's parameter name, so keyword calls work too.
    def get_service_client(service_client_type: type, *args: Any, **kwargs: Any) -> Any:
        if service_client_type in (DatabaseManagerClient, DatabaseManagerAsyncClient):
            return _FakeDatabaseManagerClient(service_client_type)
        return real_get_service_client(service_client_type, *args, **kwargs)

    getters = (
        clients.get_database_manager_client,
        clients.get_database_manager_async_client,
    )
    for getter in getters:
        clear_thread_cache(getter)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(service, "get_service_client", get_service_client)
        yield
    for getter in getters:
        clear_thread_cache(getter)


@pytest.fixture(autouse=True)
def database_manager(fake_database_manager_clients: None) -> DatabaseManagerFake:
    """A fresh record of the calls the fake answers during this test."""
    global _current
    _current = DatabaseManagerFake()
    return _current


def write_calls_summary(terminalreporter: Any) -> None:
    """List the calls the fake answered in this session, so a CI log shows
    which code paths still reach for a database manager instead of a stub of
    their own."""
    if not _CALLS_THIS_SESSION:
        return
    terminalreporter.write_sep("-", "database manager calls answered by the fake")
    for name, count in _CALLS_THIS_SESSION.most_common():
        terminalreporter.write_line(f"{count:6d}  {name}")
