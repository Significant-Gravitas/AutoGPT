"""Access-control tests for the scoped (org/team) credential store.

These lock the in-function authz rules — the store must not trust
callers to have verified team membership before handing out secrets.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.integrations import scoped_credentials

ORG_ID = "org-1"
TEAM_A = "team-a"
TEAM_B = "team-b"
USER_ID = "user-1"
OTHER_USER = "user-2"


def _cred(
    *,
    owner_type: str,
    owner_id: str,
    organization_id: str = ORG_ID,
    created_by: str = USER_ID,
):
    cred = MagicMock()
    cred.id = "cred-1"
    cred.organizationId = organization_id
    cred.ownerType = owner_type
    cred.ownerId = owner_id
    cred.createdByUserId = created_by
    cred.provider = "github"
    cred.credentialType = "api_key"
    cred.displayName = "GitHub"
    cred.lastUsedAt = None
    cred.expiresAt = None
    cred.createdAt = None
    cred.encryptedPayload = "enc"
    return cred


@pytest.fixture
def mock_prisma(mocker):
    p = MagicMock()
    mocker.patch.object(scoped_credentials, "prisma", p)
    return p


@pytest.mark.asyncio
async def test_team_cred_denied_outside_owning_team(mock_prisma):
    """Regression: an org member NOT in the owning team must not fetch a
    team credential by ID — previously the function trusted callers to
    have verified membership, and nothing did."""
    mock_prisma.integrationcredential.find_unique = AsyncMock(
        return_value=_cred(owner_type="TEAM", owner_id=TEAM_B)
    )
    mock_prisma.teammember.find_unique = AsyncMock(return_value=None)

    result = await scoped_credentials.get_credential_by_id(
        "cred-1", user_id=USER_ID, organization_id=ORG_ID, team_id=TEAM_A
    )

    assert result is None


@pytest.mark.asyncio
async def test_team_cred_allowed_for_active_team(mock_prisma):
    mock_prisma.integrationcredential.find_unique = AsyncMock(
        return_value=_cred(owner_type="TEAM", owner_id=TEAM_A)
    )

    result = await scoped_credentials.get_credential_by_id(
        "cred-1", user_id=USER_ID, organization_id=ORG_ID, team_id=TEAM_A
    )

    assert result is not None
    assert result["scope"] == "TEAM"


@pytest.mark.asyncio
async def test_team_cred_allowed_via_verified_membership(mock_prisma):
    """A member of the owning team may fetch its credential even when their
    ACTIVE context is a different team."""
    mock_prisma.integrationcredential.find_unique = AsyncMock(
        return_value=_cred(owner_type="TEAM", owner_id=TEAM_B)
    )
    membership = MagicMock()
    membership.status = "ACTIVE"
    mock_prisma.teammember.find_unique = AsyncMock(return_value=membership)

    result = await scoped_credentials.get_credential_by_id(
        "cred-1", user_id=USER_ID, organization_id=ORG_ID, team_id=TEAM_A
    )

    assert result is not None
    mock_prisma.teammember.find_unique.assert_awaited_once_with(
        where={"teamId_userId": {"teamId": TEAM_B, "userId": USER_ID}}
    )


@pytest.mark.asyncio
async def test_user_cred_denied_for_other_user(mock_prisma):
    mock_prisma.integrationcredential.find_unique = AsyncMock(
        return_value=_cred(
            owner_type="USER", owner_id=OTHER_USER, created_by=OTHER_USER
        )
    )

    result = await scoped_credentials.get_credential_by_id(
        "cred-1", user_id=USER_ID, organization_id=ORG_ID
    )

    assert result is None


@pytest.mark.asyncio
async def test_wrong_org_denied(mock_prisma):
    mock_prisma.integrationcredential.find_unique = AsyncMock(
        return_value=_cred(
            owner_type="ORG", owner_id="org-OTHER", organization_id="org-OTHER"
        )
    )

    result = await scoped_credentials.get_credential_by_id(
        "cred-1", user_id=USER_ID, organization_id=ORG_ID
    )

    assert result is None


@pytest.mark.asyncio
async def test_delete_denied_for_non_creator(mock_prisma):
    """Regression: any org member could revoke anyone's credential — the
    'admin check done at route level' comment guarded nothing."""
    mock_prisma.integrationcredential.find_unique = AsyncMock(
        return_value=_cred(
            owner_type="USER", owner_id=OTHER_USER, created_by=OTHER_USER
        )
    )
    mock_prisma.integrationcredential.update = AsyncMock()

    with pytest.raises(ValueError):
        await scoped_credentials.delete_credential(
            "cred-1", user_id=USER_ID, organization_id=ORG_ID
        )
    mock_prisma.integrationcredential.update.assert_not_called()


@pytest.mark.asyncio
async def test_delete_allowed_for_creator(mock_prisma):
    mock_prisma.integrationcredential.find_unique = AsyncMock(
        return_value=_cred(owner_type="USER", owner_id=USER_ID, created_by=USER_ID)
    )
    mock_prisma.integrationcredential.update = AsyncMock()

    await scoped_credentials.delete_credential(
        "cred-1", user_id=USER_ID, organization_id=ORG_ID
    )
    mock_prisma.integrationcredential.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_allowed_for_org_admin(mock_prisma):
    mock_prisma.integrationcredential.find_unique = AsyncMock(
        return_value=_cred(owner_type="ORG", owner_id=ORG_ID, created_by=OTHER_USER)
    )
    mock_prisma.integrationcredential.update = AsyncMock()

    await scoped_credentials.delete_credential(
        "cred-1", user_id=USER_ID, organization_id=ORG_ID, is_org_admin=True
    )
    mock_prisma.integrationcredential.update.assert_awaited_once()


# --------------------- TEAM CREDENTIAL WRITES --------------------- #


@pytest.mark.asyncio
async def test_create_team_credential_writes_expected_row_shape(mock_prisma, mocker):
    """The written row must match exactly what the read path resolves on:
    ownerType=TEAM, ownerId=teamId, teamId set, organizationId=team's org."""
    mocker.patch.object(
        scoped_credentials._cryptor, "encrypt", return_value="encrypted-blob"
    )
    created = _cred(owner_type="TEAM", owner_id=TEAM_A)
    mock_prisma.integrationcredential.create = AsyncMock(return_value=created)

    result = await scoped_credentials.create_credential(
        organization_id=ORG_ID,
        owner_type="TEAM",
        owner_id=TEAM_A,
        team_id=TEAM_A,
        provider="github",
        credential_type="api_key",
        display_name="Team GitHub",
        payload={"type": "api_key", "api_key": "secret"},
        user_id=USER_ID,
    )

    data = mock_prisma.integrationcredential.create.await_args.kwargs["data"]
    assert data["ownerType"] == "TEAM"
    assert data["ownerId"] == TEAM_A
    assert data["teamId"] == TEAM_A
    assert data["organizationId"] == ORG_ID
    assert data["createdByUserId"] == USER_ID
    assert data["encryptedPayload"] == "encrypted-blob"
    assert result["scope"] == "TEAM"


@pytest.mark.asyncio
async def test_list_team_credentials_scopes_query_to_team(mock_prisma):
    mock_prisma.integrationcredential.find_many = AsyncMock(
        return_value=[_cred(owner_type="TEAM", owner_id=TEAM_A)]
    )

    result = await scoped_credentials.list_team_credentials(ORG_ID, TEAM_A)

    where = mock_prisma.integrationcredential.find_many.await_args.kwargs["where"]
    assert where["organizationId"] == ORG_ID
    assert where["ownerType"] == "TEAM"
    assert where["ownerId"] == TEAM_A
    assert where["status"] == "active"
    assert result[0]["scope"] == "TEAM"


@pytest.mark.asyncio
async def test_delete_team_credential_revokes_own_team_cred(mock_prisma):
    mock_prisma.integrationcredential.find_unique = AsyncMock(
        return_value=_cred(owner_type="TEAM", owner_id=TEAM_A)
    )
    mock_prisma.integrationcredential.update = AsyncMock()

    await scoped_credentials.delete_team_credential(
        "cred-1", team_id=TEAM_A, organization_id=ORG_ID
    )

    mock_prisma.integrationcredential.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_team_credential_rejects_other_teams_cred(mock_prisma):
    """A team admin of TEAM_A must not be able to revoke TEAM_B's credential
    by ID (cross-team escalation within a shared org)."""
    mock_prisma.integrationcredential.find_unique = AsyncMock(
        return_value=_cred(owner_type="TEAM", owner_id=TEAM_B)
    )
    mock_prisma.integrationcredential.update = AsyncMock()

    with pytest.raises(ValueError):
        await scoped_credentials.delete_team_credential(
            "cred-1", team_id=TEAM_A, organization_id=ORG_ID
        )
    mock_prisma.integrationcredential.update.assert_not_called()


@pytest.mark.asyncio
async def test_delete_team_credential_rejects_non_team_cred(mock_prisma):
    """The team-scoped delete must never touch a USER- or ORG-owned row."""
    mock_prisma.integrationcredential.find_unique = AsyncMock(
        return_value=_cred(owner_type="USER", owner_id=USER_ID)
    )
    mock_prisma.integrationcredential.update = AsyncMock()

    with pytest.raises(ValueError):
        await scoped_credentials.delete_team_credential(
            "cred-1", team_id=TEAM_A, organization_id=ORG_ID
        )
    mock_prisma.integrationcredential.update.assert_not_called()


@pytest.mark.asyncio
async def test_delete_team_credential_rejects_wrong_org(mock_prisma):
    mock_prisma.integrationcredential.find_unique = AsyncMock(
        return_value=_cred(
            owner_type="TEAM", owner_id=TEAM_A, organization_id="org-OTHER"
        )
    )
    mock_prisma.integrationcredential.update = AsyncMock()

    with pytest.raises(ValueError):
        await scoped_credentials.delete_team_credential(
            "cred-1", team_id=TEAM_A, organization_id=ORG_ID
        )
    mock_prisma.integrationcredential.update.assert_not_called()
