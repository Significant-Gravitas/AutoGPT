"""Platform User provisioning against the real database.

The unit tests in ``user_test.py`` patch the queries out, so a flipped
comparison or a dropped filter in the SQL would leave them green.
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from prisma.models import AuthSession, AuthUser, Organization, User

from backend.data.db import execute_raw_with_schema
from backend.data.user import (
    find_orphaned_auth_identities,
    get_or_create_user_with_status,
)
from backend.util.test import SpinTestServer


async def _identity(
    created_at: datetime, *, verified: bool = True, email: str | None = None
) -> str:
    user_id = str(uuid4())
    await AuthUser.prisma().create(
        data={
            "id": user_id,
            "name": "Sweep test",
            "email": email or f"{user_id}@example.com",
            "emailVerified": verified,
            "createdAt": created_at,
        }
    )
    return user_id


@pytest.mark.asyncio(loop_scope="session")
async def test_finds_only_identities_that_can_sign_in_without_a_user_row(
    server: SpinTestServer,
):
    now = datetime.now(timezone.utc)
    old = now - timedelta(hours=1)
    cutoff = now - timedelta(minutes=5)
    owner_id = str(uuid4())
    ids: dict[str, str] = {}
    try:
        # Oldest of all, so ordering by age alone would put it first.
        ids["collided"] = await _identity(
            old - timedelta(minutes=1), email=f"Taken-{owner_id}@Example.com"
        )
        await User.prisma().create(
            data={"id": owner_id, "email": f"taken-{owner_id}@example.com"}
        )
        ids["orphan"] = await _identity(old)
        ids["provisioned"] = await _identity(old)
        await User.prisma().create(
            data={"id": ids["provisioned"], "email": f"{ids['provisioned']}@x.com"}
        )
        ids["in_grace"] = await _identity(now)
        # AUTH_REQUIRE_EMAIL_VERIFICATION on: no session until the link.
        ids["awaiting_verification"] = await _identity(old, verified=False)
        # Flag off: an unverified password user is signed in at once.
        ids["unverified_signed_in"] = await _identity(old, verified=False)
        await AuthSession.prisma().create(
            data={
                "id": str(uuid4()),
                "token": str(uuid4()),
                "userId": ids["unverified_signed_in"],
                "expiresAt": now + timedelta(days=1),
            }
        )

        found = await find_orphaned_auth_identities(cutoff, limit=1000)

        mine = [i for i in found if i.id in ids.values()]
        by_id = {i.id: i for i in mine}
        assert set(by_id) == {
            ids["orphan"],
            ids["unverified_signed_in"],
            ids["collided"],
        }
        assert by_id[ids["collided"]].email_owner_id == owner_id
        assert by_id[ids["collided"]].has_email_collision
        # Never healed, so it must not take a batch slot ahead of one that is.
        assert mine[-1].id == ids["collided"]
    finally:
        await AuthUser.prisma().delete_many(where={"id": {"in": list(ids.values())}})
        await User.prisma().delete_many(
            where={"id": {"in": [owner_id, ids.get("provisioned", owner_id)]}}
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_first_bootstrap_after_the_auth_hook_reports_the_account_created(
    server: SpinTestServer,
):
    """The auth hook inserts the bare row before the client's
    ``POST /auth/user``. That call must still answer "created" (the sign-up
    conversion header) exactly once, and only that call."""
    user_id = str(uuid4())
    payload = {"sub": user_id, "email": f"{user_id}@example.com"}
    try:
        # The exact statement in frontend/src/lib/auth/provision-platform-user.ts.
        await execute_raw_with_schema(
            'INSERT INTO {schema_prefix}"User" (id, email, name, "updatedAt") '
            "VALUES ($1, $2, $3, NOW()) ON CONFLICT (id) DO NOTHING",
            user_id,
            payload["email"],
            None,
        )

        first = await get_or_create_user_with_status(payload)
        second = await get_or_create_user_with_status(payload)

        assert first.was_created is True
        assert second.was_created is False
    finally:
        # The personal org has no FK to its User, so it would outlive the
        # User's cascade; deleting it takes its members, team and balance.
        await Organization.prisma().delete_many(where={"bootstrapUserId": user_id})
        await User.prisma().delete_many(where={"id": user_id})
