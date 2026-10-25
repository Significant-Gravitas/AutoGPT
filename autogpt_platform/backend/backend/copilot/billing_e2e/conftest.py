"""The fake provider and the platform ledger each billing scenario runs on."""

from collections.abc import AsyncIterator, Generator

import pytest
import pytest_asyncio

from .ledger import PlatformLedger, deferred_tasks
from .provider import FakeProvider


@pytest_asyncio.fixture(loop_scope="function")
async def provider() -> AsyncIterator[FakeProvider]:
    async with FakeProvider() as fake:
        yield fake


@pytest.fixture
def ledger(monkeypatch: pytest.MonkeyPatch) -> PlatformLedger:
    ledger = PlatformLedger()
    ledger.install(monkeypatch)
    return ledger


@pytest.hookimpl(wrapper=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> Generator[None, object, object]:
    """Fail a scenario that returns with its turns' deferred writes still running.

    A real turn leaves work past its own end: the cost reconcile, the cost-log
    rows, the memory ingest, each of which reaches Redis and the database. Left
    running, they are cut off when the test's event loop closes, in the middle
    of a request on a client the next test shares, and a later teardown waits on
    that client for good. ``ledger.settle()`` after each turn lets them land.
    """
    before = set(deferred_tasks())
    result = yield
    if left := set(deferred_tasks()) - before:
        pytest.fail(
            f"{pyfuncitem.nodeid} returned with {len(left)} of its turns' deferred "
            "writes still running; await ledger.settle() after each turn.",
            pytrace=False,
        )
    return result
