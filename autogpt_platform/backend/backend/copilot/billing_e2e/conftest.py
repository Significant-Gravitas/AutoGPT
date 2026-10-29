"""The fake provider and the platform ledger each billing scenario runs on."""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from .ledger import PlatformLedger
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
