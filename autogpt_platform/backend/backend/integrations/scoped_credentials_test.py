"""Access-control tests for the scoped (org/team) credential store.

These lock the in-function authz rules — the store must not trust
callers to have verified team membership before handing out secrets.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma import Json
from prisma.enums import CredentialOwnerType

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
    status: str = "active",
    metadata: dict | None = None,
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
    cred.status = status
    cred.metadata = metadata
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
async def test_revoked_cred_is_invisible_by_id(mock_prisma):
    """A soft-deleted credential must not remain readable (or decryptable) by
    ID — the list queries filter on `status`, and this path has to agree."""
    mock_prisma.integrationcredential.find_unique = AsyncMock(
        return_value=_cred(owner_type="TEAM", owner_id=TEAM_A, status="revoked")
    )

    result = await scoped_credentials.get_credential_by_id(
        "cred-1",
        user_id=USER_ID,
        organization_id=ORG_ID,
        team_id=TEAM_A,
        decrypt=True,
    )

    assert result is None


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
    ownerType=TEAM, ownerId=teamId, teamId set, organizationId=team's org.

    The relation fields must use Prisma's `connect` form: `Organization` is a
    required relation (a raw `organizationId` scalar is rejected by the query
    engine with MissingRequiredValueError) and the teamId relation is named
    `Workspace`, not `Team`.
    """
    mocker.patch.object(
        scoped_credentials,
        "_cryptor",
        MagicMock(encrypt=MagicMock(return_value="encrypted-blob")),
    )
    created = _cred(owner_type="TEAM", owner_id=TEAM_A)
    mock_prisma.integrationcredential.create = AsyncMock(return_value=created)

    result = await scoped_credentials.create_credential(
        organization_id=ORG_ID,
        owner_type=CredentialOwnerType.TEAM,
        owner_id=TEAM_A,
        provider="github",
        credential_type="api_key",
        display_name="Team GitHub",
        payload={"type": "api_key", "api_key": "secret"},
        user_id=USER_ID,
    )

    data = mock_prisma.integrationcredential.create.await_args.kwargs["data"]
    assert data["ownerType"] == CredentialOwnerType.TEAM
    assert data["ownerId"] == TEAM_A
    assert data["Organization"] == {"connect": {"id": ORG_ID}}
    assert data["Workspace"] == {"connect": {"id": TEAM_A}}
    assert "organizationId" not in data
    assert "teamId" not in data
    assert data["createdByUserId"] == USER_ID
    assert data["encryptedPayload"] == "encrypted-blob"
    assert result["scope"] == CredentialOwnerType.TEAM


@pytest.mark.asyncio
async def test_create_non_team_credential_omits_workspace_relation(mock_prisma, mocker):
    """`teamId` is only meaningful for TEAM rows; USER/ORG rows must leave the
    `Workspace` relation out entirely rather than connecting a null id."""
    mocker.patch.object(
        scoped_credentials,
        "_cryptor",
        MagicMock(encrypt=MagicMock(return_value="encrypted-blob")),
    )
    mock_prisma.integrationcredential.create = AsyncMock(
        return_value=_cred(owner_type="USER", owner_id=USER_ID)
    )

    await scoped_credentials.create_credential(
        organization_id=ORG_ID,
        owner_type=CredentialOwnerType.USER,
        owner_id=USER_ID,
        provider="github",
        credential_type="api_key",
        display_name="My GitHub",
        payload={"type": "api_key", "api_key": "secret"},
        user_id=USER_ID,
    )

    data = mock_prisma.integrationcredential.create.await_args.kwargs["data"]
    assert "Workspace" not in data
    assert data["Organization"] == {"connect": {"id": ORG_ID}}


@pytest.mark.asyncio
async def test_create_credential_wraps_metadata_as_prisma_json(mock_prisma, mocker):
    """`metadata` is a `Json?` column — a raw dict is not a valid input value,
    and passing `None` for "no metadata" is not the same as omitting the key."""
    mocker.patch.object(
        scoped_credentials,
        "_cryptor",
        MagicMock(encrypt=MagicMock(return_value="encrypted-blob")),
    )
    mock_prisma.integrationcredential.create = AsyncMock(
        return_value=_cred(owner_type="TEAM", owner_id=TEAM_A)
    )

    await scoped_credentials.create_credential(
        organization_id=ORG_ID,
        owner_type=CredentialOwnerType.TEAM,
        owner_id=TEAM_A,
        provider="github",
        credential_type="host_scoped",
        display_name="Team GitHub",
        payload={"type": "host_scoped", "host": "api.github.com"},
        user_id=USER_ID,
        metadata={"host": "api.github.com"},
    )
    data = mock_prisma.integrationcredential.create.await_args.kwargs["data"]
    assert isinstance(data["metadata"], Json)

    mock_prisma.integrationcredential.create.reset_mock()
    await scoped_credentials.create_credential(
        organization_id=ORG_ID,
        owner_type=CredentialOwnerType.TEAM,
        owner_id=TEAM_A,
        provider="github",
        credential_type="api_key",
        display_name="Team GitHub",
        payload={"type": "api_key", "api_key": "secret"},
        user_id=USER_ID,
    )
    data = mock_prisma.integrationcredential.create.await_args.kwargs["data"]
    assert "metadata" not in data


@pytest.mark.asyncio
async def test_create_credential_syncs_payload_id_with_row_id(mock_prisma, mocker):
    """The encrypted payload's id must equal the row's primary key so a
    decrypted read resolves to the same credential the row represents."""
    captured: dict = {}

    def _fake_encrypt(payload):
        captured["payload"] = payload
        return "encrypted-blob"

    mocker.patch.object(
        scoped_credentials,
        "_cryptor",
        MagicMock(encrypt=MagicMock(side_effect=_fake_encrypt)),
    )
    mock_prisma.integrationcredential.create = AsyncMock(
        return_value=_cred(owner_type="TEAM", owner_id=TEAM_A)
    )

    await scoped_credentials.create_credential(
        organization_id=ORG_ID,
        owner_type=CredentialOwnerType.TEAM,
        owner_id=TEAM_A,
        provider="github",
        credential_type="api_key",
        display_name="Team GitHub",
        payload={"id": "client-supplied-id", "type": "api_key", "api_key": "secret"},
        user_id=USER_ID,
    )

    data = mock_prisma.integrationcredential.create.await_args.kwargs["data"]
    # Row id is server-generated, not the client-supplied one ...
    assert data["id"] != "client-supplied-id"
    # ... and the encrypted payload carries that same authoritative id.
    assert captured["payload"]["id"] == data["id"]


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
async def test_delete_team_credential_scopes_the_write_itself(mock_prisma):
    """The ownership check IS the `where` of the write — a find-then-update
    would leave a window for the row to change owner between the two calls."""
    mock_prisma.integrationcredential.update_many = AsyncMock(return_value=1)

    await scoped_credentials.delete_team_credential(
        "cred-1", team_id=TEAM_A, organization_id=ORG_ID
    )

    call = mock_prisma.integrationcredential.update_many.await_args.kwargs
    assert call["where"] == {
        "id": "cred-1",
        "organizationId": ORG_ID,
        "ownerType": CredentialOwnerType.TEAM,
        "ownerId": TEAM_A,
        "status": scoped_credentials.CredentialStatus.ACTIVE,
    }
    assert call["data"] == {"status": scoped_credentials.CredentialStatus.REVOKED}


@pytest.mark.asyncio
async def test_delete_team_credential_raises_when_nothing_matched(mock_prisma):
    """Anything the scoped `where` doesn't match — another team's credential
    (cross-team escalation inside a shared org), a USER/ORG-owned row, a row in
    a different org, or an already-revoked one — must surface as not-found."""
    mock_prisma.integrationcredential.update_many = AsyncMock(return_value=0)

    with pytest.raises(ValueError):
        await scoped_credentials.delete_team_credential(
            "cred-1", team_id=TEAM_A, organization_id=ORG_ID
        )
