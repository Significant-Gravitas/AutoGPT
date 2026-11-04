"""A team plan from the sale to the day it ends, end to end.

Sales invoices the customer; an admin sets the team organization's plan and
adds the invoice's credits; the team works under that plan until its end date.
Drives the real app against the real database: the paywall every paid route
asks, the organization's seats, its wallet paying for a run, and the plan's end.
Billing is off in tests by default, so the test turns it on itself.
"""

import asyncio
import uuid
from collections.abc import Coroutine
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from autogpt_libs.auth import get_optional_user_id, get_user_id
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.models import AuditLog

from backend.api.model import CreateGraph
from backend.api.rest_api import app
from backend.copilot.rate_limit import SubscriptionTier, set_user_tier
from backend.data import credit, db_manager
from backend.data.credit import UserCredit
from backend.data.execution import GraphExecutionEntry, NodeExecutionEntry
from backend.data.org_access import has_org_access
from backend.data.user import get_or_create_user
from backend.executor import billing
from backend.executor import utils as executor_utils
from backend.usecases.sample import create_test_graph
from backend.util.settings import Config
from backend.util.test import SpinTestServer

pytestmark = pytest.mark.real_org_access

PAID_PROBE = "/api/workspace/files/{file_id}/preview"


class _Account:
    def __init__(self, user_id: str, email: str, role: str = "user") -> None:
        self.user_id = user_id
        self.email = email
        self.role = role

    def payload(self) -> dict[str, str]:
        return {"sub": self.user_id, "role": self.role, "email": self.email}


async def _account(label: str, role: str = "user") -> _Account:
    email = f"org-plan-e2e-{label}-{uuid.uuid4().hex[:8]}@example.com"
    user = await get_or_create_user(
        {"sub": str(uuid.uuid4()), "email": email, "name": label}
    )
    return _Account(user.id, email, role)


@pytest.fixture
def payments_on(monkeypatch):
    for name in (
        "FORCE_FLAG_SHOW_ORG_SETTINGS",
        "NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS",
        "FORCE_ALL_FLAGS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FORCE_FLAG_ENABLE_PLATFORM_PAYMENT", "true")


@pytest.fixture
def credits_on(monkeypatch):
    """Turn billing on where this test's charge and wallet reads check it:
    the executor's billing, and the credit ledgers (org_credit reads
    credit's settings)."""
    monkeypatch.setattr(billing.settings.config, "enable_credit", True)
    monkeypatch.setattr(credit.settings.config, "enable_credit", True)


@pytest.fixture
async def client():
    previous = dict(app.dependency_overrides)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    app.dependency_overrides.update(previous)


async def _as(
    client: httpx.AsyncClient,
    account: _Account,
    method: str,
    url: str,
    org_id: str | None = None,
    **kwargs,
) -> httpx.Response:
    app.dependency_overrides[get_jwt_payload] = account.payload
    app.dependency_overrides[get_user_id] = lambda: account.user_id
    app.dependency_overrides[get_optional_user_id] = lambda: account.user_id
    has_org_access.cache_delete(account.user_id)
    headers = {"X-Org-Id": org_id} if org_id else {}
    return await client.request(method, url, headers=headers, **kwargs)


async def _paid_status(
    client: httpx.AsyncClient, account: _Account, org_id: str | None
) -> int:
    """402 when the paywall refuses; 404 (no such file) once it lets through."""
    url = PAID_PROBE.format(file_id=uuid.uuid4())
    return (await _as(client, account, "GET", url, org_id)).status_code


@pytest.mark.asyncio(loop_scope="session")
async def test_a_team_plan_from_sale_to_its_end_date(
    server: SpinTestServer,
    payments_on,
    credits_on,
    real_entitlement,
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
):
    owner = await _account("owner")
    invitee = await _account("invitee")
    admin = await _account("admin", role="admin")
    await set_user_tier(owner.user_id, SubscriptionTier.PRO)
    enabled = await _as(
        client, admin, "PUT", f"/api/admin/org-access/users/{owner.user_id}", json={}
    )
    assert enabled.status_code == 200, enabled.text

    # 1. The owner's personal Pro covers their personal org, not the team org.
    slug = f"acme-{uuid.uuid4().hex[:8]}"
    created = await _as(
        client, owner, "POST", "/api/orgs", json={"name": "Acme", "slug": slug}
    )
    assert created.status_code == 200, created.text
    org_id = created.json()["id"]
    refused = await _as(
        client, owner, "GET", PAID_PROBE.format(file_id=uuid.uuid4()), org_id
    )
    assert refused.status_code == 402
    assert "organization needs an active plan" in refused.json()["detail"]
    assert await _paid_status(client, owner, None) == 404

    # 2. An admin sets a plan with three seats, and the owner gets through.
    plan = await _as(
        client,
        admin,
        "PUT",
        f"/api/admin/org-plans/orgs/{org_id}",
        json={"tier": "BUSINESS", "seats": 3, "note": "INV-1001"},
    )
    assert plan.status_code == 200, plan.text
    assert plan.json()["plan"]["in_effect"] is True
    assert await _paid_status(client, owner, org_id) == 404

    # 3. Two invitations fit beside the owner; a third finds no free seat.
    tokens = []
    for email in (invitee.email, f"second-{slug}@example.com"):
        invited = await _as(
            client,
            owner,
            "POST",
            f"/api/orgs/{org_id}/invitations",
            org_id,
            json={"email": email},
        )
        assert invited.status_code == 200, invited.text
        tokens.append(invited.json()["token"])
    full = await _as(
        client,
        owner,
        "POST",
        f"/api/orgs/{org_id}/invitations",
        org_id,
        json={"email": f"third-{slug}@example.com"},
    )
    assert full.status_code == 409
    assert "seats" in full.json()["detail"]

    # 4. The invitee, with no plan of their own, joins and works under the
    # team plan, and stays paywalled in their own personal org.
    accepted = await _as(
        client, invitee, "POST", f"/api/invitations/{tokens[0]}/accept"
    )
    assert accepted.status_code == 200, accepted.text
    assert await _paid_status(client, invitee, org_id) == 404
    assert await _paid_status(client, invitee, None) == 402

    # 5. Credits keyed by the invoice are added once, however often they're sent.
    for granted in (True, False):
        credits = await _as(
            client,
            admin,
            "POST",
            f"/api/admin/org-plans/orgs/{org_id}/credits",
            json={"amount": 1000, "reference": "INV-1001"},
        )
        assert credits.status_code == 200, credits.text
        assert credits.json()["granted"] is granted
        assert credits.json()["balance"] == 1000

    # 6. A run in the team org is paid from the org's wallet, not the owner's.
    owner_wallet = await UserCredit().get_credits(owner.user_id)
    charged = await _charge_a_team_run(client, owner, org_id, monkeypatch)
    assert charged > 0
    assert await UserCredit().get_credits(owner.user_id) == owner_wallet
    # The admin finds the org by its owner's email: two members and the one
    # invitation still pending hold its three seats, and its wallet paid.
    listed = await _as(
        client, admin, "GET", "/api/admin/org-plans/orgs", params={"email": owner.email}
    )
    assert listed.status_code == 200, listed.text
    (summary,) = [org for org in listed.json() if org["org_id"] == org_id]
    assert (summary["members"], summary["pending_invitations"]) == (2, 1)
    assert summary["seats_in_use"] == 3
    assert summary["balance"] == 1000 - charged

    # 7. Past its end date the plan stops counting, for everyone in the org.
    ended = await _as(
        client,
        admin,
        "PUT",
        f"/api/admin/org-plans/orgs/{org_id}",
        json={
            "tier": "BUSINESS",
            "seats": 3,
            "ends_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        },
    )
    assert ended.status_code == 200, ended.text
    assert ended.json()["plan"]["in_effect"] is False
    assert await _paid_status(client, owner, org_id) == 402
    assert await _paid_status(client, invitee, org_id) == 402

    # Each admin change left an audit row.
    actions = [
        row.action
        for row in await AuditLog.prisma().find_many(
            where={"organizationId": org_id, "actorUserId": admin.user_id},
            order={"createdAt": "asc"},
        )
    ]
    assert actions == ["org_plan.set", "org_credits.grant", "org_plan.set"]


class _InProcessBillingDb:
    """Billing's database client, answered in this process.

    The executor's billing asks the database manager, a separate process
    that reads the billing switch from its own environment. This test turns
    billing on here instead (``credits_on``), so the ledger calls run here:
    billing runs in a worker thread, and each call runs on the test's loop,
    where the database client lives."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _run(self, call: Coroutine[Any, Any, int]) -> int:
        return asyncio.run_coroutine_threadsafe(call, self._loop).result()

    def spend_org_credits(self, **kwargs: Any) -> int:
        return self._run(db_manager.spend_org_credits(**kwargs))

    def spend_credits(self, **kwargs: Any) -> int:
        return self._run(db_manager.spend_credits(**kwargs))

    def get_org_credits(self, **kwargs: Any) -> int:
        return self._run(db_manager.get_org_credits(**kwargs))

    def get_credits(self, **kwargs: Any) -> int:
        return self._run(db_manager.get_credits(**kwargs))


async def _charge_a_team_run(
    client: httpx.AsyncClient,
    owner: _Account,
    org_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> int:
    """Start a run in the team org through the API, then charge one of its
    node executions the way the executor does; returns what was charged.

    The run is taken off the queue so the shared test executor doesn't run it;
    its queued context is what billing reads to pick the wallet."""
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(billing, "get_db_client", lambda: _InProcessBillingDb(loop))
    queued: list[str] = []

    class _Queue:
        async def publish_message(self, *, message: str, **_: object) -> None:
            queued.append(message)

    async def _execution_queue() -> _Queue:
        return _Queue()

    monkeypatch.setattr(executor_utils, "get_async_execution_queue", _execution_queue)
    graph = CreateGraph(graph=create_test_graph()).model_dump(mode="json")
    created = await _as(client, owner, "POST", "/api/graphs", org_id, json=graph)
    assert created.status_code == 200, created.text
    graph_id, version = created.json()["id"], created.json()["version"]
    started = await _as(
        client,
        owner,
        "POST",
        f"/api/graphs/{graph_id}/execute/{version}",
        org_id,
        json={"inputs": {"input_1": "a", "input_2": "b"}},
    )
    assert started.status_code == 200, started.text
    (message,) = queued
    entry = GraphExecutionEntry.model_validate_json(message)
    assert entry.execution_context.organization_id == org_id
    node = created.json()["nodes"][-1]
    node_exec = NodeExecutionEntry(
        user_id=owner.user_id,
        graph_exec_id=entry.graph_exec_id,
        graph_id=graph_id,
        graph_version=version,
        node_exec_id=str(uuid.uuid4()),
        node_id=node["id"],
        block_id=node["block_id"],
        inputs={},
        execution_context=entry.execution_context,
    )
    # Every N-th node execution of a run costs credits: charge that one.
    total, _, _ = await asyncio.to_thread(
        billing.charge_usage, node_exec, Config().execution_cost_count_threshold
    )
    return total
