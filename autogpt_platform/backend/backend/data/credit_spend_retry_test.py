"""A spend is charged once even when a step after its debit fails.

The database manager's client retries a call that fails, so a failure after
the debit is committed must not fail the call. These spend through the
manager's own app behind that retrying client, against the test database,
with one database call after the debit failing once.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, ParamSpec, TypeVar, cast
from uuid import uuid4

import httpx
import pytest
from prisma.actions import OrgCreditTransactionActions
from prisma.enums import CreditTransactionType
from prisma.errors import PrismaError
from prisma.models import (
    CreditTransaction,
    Organization,
    OrgBalance,
    OrgCreditTransaction,
    User,
    UserBalance,
)

from backend.data import credit
from backend.data.credit import UsageTransactionMetadata, UserCredit
from backend.data.db_manager import DatabaseManager, DatabaseManagerAsyncClient
from backend.data.model import AutoTopUpConfig
from backend.data.org_credit import get_org_credits
from backend.util import service as service_module
from backend.util.json import SafeJson
from backend.util.service import get_service_client
from backend.util.service_auth import internal_auth_headers

P = ParamSpec("P")
R = TypeVar("R")

SPEND = UsageTransactionMetadata(
    graph_exec_id="graph-exec-1", graph_id="graph-1", node_exec_id="node-exec-1"
)


async def test_a_spend_is_charged_once_when_the_top_up_check_fails(
    user_id: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(credit.settings.config, "enable_credit", True)
    monkeypatch.setattr(credit, "get_auto_top_up", _fails_once(credit.get_auto_top_up))

    async with _database_manager(monkeypatch) as client:
        returned = await client.spend_credits(user_id=user_id, cost=100, metadata=SPEND)

    usage = await CreditTransaction.prisma().find_many(
        where={"userId": user_id, "type": CreditTransactionType.USAGE}
    )
    assert {
        "returned": returned,
        "balance": await UserCredit().get_credits(user_id),
        "usage": [row.amount for row in usage],
    } == {"returned": 900, "balance": 900, "usage": [-100]}


async def test_a_team_spend_is_charged_once_when_its_ledger_write_fails(
    team_org_id: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        OrgCreditTransactionActions,
        "create",
        _fails_once(OrgCreditTransactionActions.create),
    )

    async with _database_manager(monkeypatch) as client:
        returned = await client.spend_org_credits(
            org_id=team_org_id, user_id="member-1", cost=100, metadata=SPEND
        )

    assert {
        "returned": returned,
        "balance": await get_org_credits(team_org_id),
    } == {"returned": 900, "balance": 900}


@pytest.fixture
async def user_id(server) -> AsyncIterator[str]:
    """A user with 1,000 credits and an auto top-up a 100-credit spend
    doesn't reach."""
    user_id = f"spend-retry-{uuid4()}"
    top_up = AutoTopUpConfig(threshold=500, amount=1000)
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"{user_id}@example.com",
            "topUpConfig": SafeJson(top_up.model_dump()),
        }
    )
    await UserBalance.prisma().create(data={"userId": user_id, "balance": 1000})
    yield user_id
    await CreditTransaction.prisma().delete_many(where={"userId": user_id})
    await User.prisma().delete_many(where={"id": user_id})


@pytest.fixture
async def team_org_id(server) -> AsyncIterator[str]:
    """A team organization with 1,000 credits."""
    org_id = str(uuid4())
    await Organization.prisma().create(
        data={"id": org_id, "name": "Spend retry", "slug": f"spend-retry-{org_id}"}
    )
    await OrgBalance.prisma().create(data={"orgId": org_id, "balance": 1000})
    yield org_id
    await OrgCreditTransaction.prisma().delete_many(where={"orgId": org_id})
    await Organization.prisma().delete_many(where={"id": org_id})


def _fails_once(call: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """*call*, except that its first call raises a database error."""
    failed = False

    async def fails_once(*args: P.args, **kwargs: P.kwargs) -> R:
        nonlocal failed
        if not failed:
            failed = True
            raise PrismaError("Server has closed the connection.")
        return await call(*args, **kwargs)

    return fails_once


@asynccontextmanager
async def _database_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[DatabaseManagerAsyncClient]:
    """The database manager's retrying client, answered in process by the
    manager's own app, which answers an error it didn't map with a 500."""
    monkeypatch.setattr(service_module, "instrument_fastapi", lambda *_a, **_k: None)
    client = get_service_client(DatabaseManagerAsyncClient, request_retry=True)
    inner = cast(Any, client)
    inner._async_clients[asyncio.get_running_loop()] = httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=DatabaseManager().build_fastapi_app(), raise_app_exceptions=False
        ),
        base_url=inner.base_url,
        headers=internal_auth_headers(),
    )
    try:
        yield client
    finally:
        await inner.aclose()
