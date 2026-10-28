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
