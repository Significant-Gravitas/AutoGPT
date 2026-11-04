"""Credits keyed by a reference are added once, even when the same grant is
sent twice at once. Runs against the database."""

import asyncio
import uuid

import pytest
from prisma.models import OrgBalance

from backend.api.features.orgs.db import create_org
from backend.data import org_plan_admin
from backend.data.user import get_or_create_user
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_two_grants_with_one_reference_add_the_credits_once(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    owner = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"grant-race-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Owner",
        }
    )
    org = await create_org("Race", f"race-{uuid.uuid4().hex[:8]}", owner.id)

    # Both grants check the reference, then wait for each other before adding.
    # The wallet lock keeps the second from checking until the first is done,
    # so the first waits out the barrier alone and the second sees its grant.
    barrier = asyncio.Barrier(2)
    checked = org_plan_admin._granted_before

    async def check_then_wait(tx, org_id: str, key: str) -> bool:
        used = await checked(tx, org_id, key)
        try:
            await asyncio.wait_for(barrier.wait(), timeout=2)
        except (TimeoutError, asyncio.BrokenBarrierError):
            pass
        return used

    monkeypatch.setattr(org_plan_admin, "_granted_before", check_then_wait)

    results = await asyncio.gather(
        *(
            org_plan_admin.grant_org_credits(org.id, owner.id, 100, "INV-RACE", None)
            for _ in range(2)
        )
    )

    assert sorted(result.granted for result in results) == [False, True]
    wallet = await OrgBalance.prisma().find_unique(where={"orgId": org.id})
    assert wallet is not None and wallet.balance == 100
