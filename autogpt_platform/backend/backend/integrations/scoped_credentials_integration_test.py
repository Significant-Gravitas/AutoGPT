"""Integration tests for the scoped credential store.

These run real database operations against the ``IntegrationCredential`` table.
Every other test in ``scoped_credentials_test.py`` mocks Prisma, so it can only
assert the *shape* of the query input — it cannot tell a shape the query engine
accepts from one it rejects. That gap hid a create path that raised
``MissingRequiredValueError`` and wrote zero rows in production:

* ``Organization`` is a **required** relation, so passing the raw
  ``organizationId`` scalar is not enough — it must be given in ``connect`` form.
* The ``teamId`` relation is named ``Workspace`` (not ``Team``).
* ``metadata`` is a ``Json?`` column, which rejects a plain Python dict.

A single real create → list → delete round trip covers all three.
"""

import uuid

import pytest
from prisma.enums import CredentialOwnerType
from prisma.models import IntegrationCredential as PrismaCredential
from prisma.models import Organization as PrismaOrg
from prisma.models import Team as PrismaTeam
from prisma.models import User as PrismaUser

from backend.integrations import scoped_credentials
from backend.util.json import SafeJson

HOST = "api.example.com"


@pytest.fixture
async def team_context():
    """Create a throw-away user + org + team, and clean them up afterwards."""
    suffix = uuid.uuid4().hex[:12]
    user_id = str(uuid.uuid4())
    org_id = str(uuid.uuid4())
    team_id = str(uuid.uuid4())

    await PrismaUser.prisma().create(
        data={
            "id": user_id,
            "email": f"scoped-creds-{suffix}@example.com",
            "topUpConfig": SafeJson({}),
            "timezone": "UTC",
        }
    )
    await PrismaOrg.prisma().create(
        data={
            "id": org_id,
            "name": f"Scoped Creds Org {suffix}",
            "slug": f"scoped-creds-{suffix}",
            "settings": SafeJson({}),
        }
    )
    await PrismaTeam.prisma().create(
        data={
            "id": team_id,
            "name": f"Scoped Creds Team {suffix}",
            "Org": {"connect": {"id": org_id}},
        }
    )

    yield org_id, team_id, user_id

    await PrismaCredential.prisma().delete_many(where={"organizationId": org_id})
    await PrismaTeam.prisma().delete_many(where={"orgId": org_id})
    await PrismaOrg.prisma().delete_many(where={"id": org_id})
    await PrismaUser.prisma().delete_many(where={"id": user_id})


@pytest.mark.asyncio(loop_scope="session")
async def test_team_credential_create_list_delete_round_trip(team_context):
    """The full lifecycle a team admin drives, against the real table."""
    org_id, team_id, user_id = team_context

    created = await scoped_credentials.create_credential(
        organization_id=org_id,
        owner_type=CredentialOwnerType.TEAM,
        owner_id=team_id,
        provider="github",
        credential_type="host_scoped",
        display_name="Team GitHub",
        payload={"type": "host_scoped", "host": HOST, "headers": {}},
        user_id=user_id,
        metadata={"host": HOST},
    )

    # The row actually exists, and carries the tenancy the read path resolves on.
    row = await PrismaCredential.prisma().find_unique(where={"id": created["id"]})
    assert row is not None
    assert row.organizationId == org_id
    # `teamId` is derived from `owner_id` through the `Workspace` relation.
    assert row.teamId == team_id
    assert row.ownerType == CredentialOwnerType.TEAM
    assert row.ownerId == team_id
    assert row.status == scoped_credentials.CredentialStatus.ACTIVE
    assert row.metadata == {"host": HOST}

    # The host survives into the list response's source data, so a host-scoped
    # credential stays distinguishable without decrypting anything.
    listed = await scoped_credentials.list_team_credentials(org_id, team_id)
    assert [c["id"] for c in listed] == [created["id"]]
    assert listed[0]["metadata"] == {"host": HOST}
    assert listed[0]["displayName"] == "Team GitHub"

    assert (
        await scoped_credentials.get_credential_by_id(
            created["id"],
            user_id=user_id,
            organization_id=org_id,
            team_id=team_id,
        )
        is not None
    )

    await scoped_credentials.delete_team_credential(created["id"], team_id, org_id)

    # Soft delete: the row survives as `revoked` but is gone from every read.
    revoked = await PrismaCredential.prisma().find_unique(where={"id": created["id"]})
    assert revoked is not None
    assert revoked.status == scoped_credentials.CredentialStatus.REVOKED
    assert await scoped_credentials.list_team_credentials(org_id, team_id) == []
    assert (
        await scoped_credentials.get_credential_by_id(
            created["id"],
            user_id=user_id,
            organization_id=org_id,
            team_id=team_id,
        )
        is None
    )

    # Revoking twice is not a silent success.
    with pytest.raises(ValueError):
        await scoped_credentials.delete_team_credential(created["id"], team_id, org_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_team_credential_delete_is_scoped_to_the_owning_team(team_context):
    """A second team in the same org must not be able to revoke team A's row."""
    org_id, team_id, user_id = team_context
    other_team = await PrismaTeam.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "name": f"Other Team {uuid.uuid4().hex[:8]}",
            "Org": {"connect": {"id": org_id}},
        }
    )

    created = await scoped_credentials.create_credential(
        organization_id=org_id,
        owner_type=CredentialOwnerType.TEAM,
        owner_id=team_id,
        provider="github",
        credential_type="api_key",
        display_name="Team GitHub",
        payload={"type": "api_key", "api_key": "secret"},
        user_id=user_id,
    )

    with pytest.raises(ValueError):
        await scoped_credentials.delete_team_credential(
            created["id"], other_team.id, org_id
        )

    still_active = await PrismaCredential.prisma().find_unique(
        where={"id": created["id"]}
    )
    assert still_active is not None
    assert still_active.status == scoped_credentials.CredentialStatus.ACTIVE
    # ... and the other team can't see it either.
    assert await scoped_credentials.list_team_credentials(org_id, other_team.id) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_user_scoped_credential_omits_team_fk(team_context):
    """USER-owned rows must leave `teamId` null — the FK is TEAM-only."""
    org_id, _team_id, user_id = team_context

    created = await scoped_credentials.create_credential(
        organization_id=org_id,
        owner_type=CredentialOwnerType.USER,
        owner_id=user_id,
        provider="github",
        credential_type="api_key",
        display_name="My GitHub",
        payload={"type": "api_key", "api_key": "secret"},
        user_id=user_id,
    )

    row = await PrismaCredential.prisma().find_unique(where={"id": created["id"]})
    assert row is not None
    assert row.teamId is None
    assert row.metadata is None
