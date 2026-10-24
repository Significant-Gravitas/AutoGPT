"""Workload tokens: minted hashed, and only accepted while current and active."""

import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import APIKeyPermission, WorkloadIdentityStatus
from prisma.models import WorkloadAccessToken as PrismaWorkloadAccessToken
from prisma.models import WorkloadIdentity as PrismaWorkloadIdentity
from pydantic import ValidationError

from backend.data.auth.workload_identity import (
    WORKLOAD_TOKEN_PREFIX,
    InvalidWorkloadTokenError,
    WorkloadIdentityInfo,
    WorkloadIdentitySpec,
    mint_workload_token,
    revoke_workload_identity,
    set_workload_identity_status,
    validate_workload_token,
)

_NOW = datetime.now(timezone.utc)


@pytest.fixture
def token_client(mocker):
    client = AsyncMock()
    mocker.patch.object(PrismaWorkloadAccessToken, "prisma", return_value=client)
    return client


@pytest.fixture
def identity_client(mocker):
    client = AsyncMock()
    mocker.patch.object(PrismaWorkloadIdentity, "prisma", return_value=client)
    return client


@pytest.mark.asyncio
async def test_a_minted_token_is_stored_only_as_a_hash(token_client):
    token_client.create.side_effect = lambda data: _token_row(
        token=data["token"], scopes=data["scopes"], expires_at=data["expiresAt"]
    )
    minted_at = datetime.now(timezone.utc)

    info, plaintext = await mint_workload_token(
        WorkloadIdentityInfo.from_db(_identity_row()),
        subject="repo:acme/api:ref:refs/heads/main",
        subject_jti="jti-1",
        scopes=[APIKeyPermission.READ_GRAPH],
    )

    data = token_client.create.await_args.kwargs["data"]
    assert plaintext.startswith(WORKLOAD_TOKEN_PREFIX)
    assert data["token"] == hashlib.sha256(plaintext.encode()).hexdigest()
    assert plaintext not in str(data)
    lifetime = data["expiresAt"] - minted_at
    assert timedelta(seconds=899) < lifetime < timedelta(seconds=910)
    assert info.type == "workload"
    assert info.user_id == "user-1"
    assert info.workload_identity_id == "wi-1"


@pytest.mark.asyncio
async def test_a_current_token_of_an_active_identity_authorizes_its_owner(
    token_client,
):
    token_client.find_unique.return_value = _token_row()

    info = await validate_workload_token("agpt_wt_plain")

    assert info.user_id == "user-1"
    assert info.scopes == [APIKeyPermission.READ_GRAPH]
    lookup = token_client.find_unique.await_args.kwargs["where"]["token"]
    assert lookup == hashlib.sha256(b"agpt_wt_plain").hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row, message",
    [
        (None, "not found"),
        (lambda: _token_row(revoked_at=_NOW), "revoked"),
        (lambda: _token_row(expires_at=_NOW - timedelta(seconds=1)), "expired"),
        (
            lambda: _token_row(identity_status=WorkloadIdentityStatus.DISABLED),
            "not active",
        ),
        (
            lambda: _token_row(identity_status=WorkloadIdentityStatus.REVOKED),
            "not active",
        ),
    ],
    ids=["unknown", "revoked", "expired", "identity disabled", "identity revoked"],
)
async def test_other_tokens_are_refused(token_client, row, message):
    token_client.find_unique.return_value = row() if row else None

    with pytest.raises(InvalidWorkloadTokenError, match=message):
        await validate_workload_token("agpt_wt_plain")


@pytest.mark.asyncio
async def test_revoking_an_identity_revokes_its_outstanding_tokens(
    identity_client, token_client
):
    identity_client.find_first.return_value = _identity_row()
    identity_client.update.return_value = _identity_row(
        status=WorkloadIdentityStatus.REVOKED
    )

    await revoke_workload_identity("wi-1", "user-1")

    token_client.update_many.assert_awaited_once()
    where = token_client.update_many.await_args.kwargs["where"]
    assert where == {"workloadIdentityId": "wi-1", "revokedAt": None}


@pytest.mark.asyncio
async def test_a_revoked_identity_cannot_be_enabled_again(identity_client):
    identity_client.find_first.return_value = _identity_row(
        status=WorkloadIdentityStatus.REVOKED
    )

    with pytest.raises(ValueError, match="revoked"):
        await set_workload_identity_status(
            "wi-1", "user-1", WorkloadIdentityStatus.ACTIVE
        )
    identity_client.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_identities_are_only_found_among_the_users_own(identity_client):
    identity_client.find_first.return_value = None

    with pytest.raises(ValueError, match="not found"):
        await revoke_workload_identity("wi-1", "someone-else", "org-1")

    where = identity_client.find_first.await_args.kwargs["where"]
    assert where["userId"] == "someone-else"
    assert where["OR"] == [{"organizationId": "org-1"}, {"organizationId": None}]


def test_a_spec_that_would_trust_any_github_repository_is_invalid():
    with pytest.raises(ValidationError, match="every account"):
        WorkloadIdentitySpec(
            name="deploy",
            issuer="https://token.actions.githubusercontent.com",
            audience="https://backend.agpt.co",
            claim_conditions=[{"claim": "ref", "pattern": "refs/heads/main"}],
            permissions=[APIKeyPermission.READ_GRAPH],
        )


def _identity_row(status=WorkloadIdentityStatus.ACTIVE) -> MagicMock:
    row = MagicMock(spec=PrismaWorkloadIdentity)
    row.id = "wi-1"
    row.name = "deploy"
    row.description = None
    row.issuer = "https://token.actions.githubusercontent.com"
    row.audience = "https://backend.agpt.co"
    row.claimConditions = [{"claim": "repository", "pattern": "acme/api"}]
    row.permissions = [APIKeyPermission.READ_GRAPH]
    row.tokenTtlSeconds = 900
    row.status = status
    row.createdAt = _NOW
    row.lastUsedAt = None
    row.revokedAt = None
    row.userId = "user-1"
    row.organizationId = None
    return row


def _token_row(
    token: str = "hash",
    scopes: list | None = None,
    expires_at: datetime | None = None,
    revoked_at: datetime | None = None,
    identity_status=WorkloadIdentityStatus.ACTIVE,
) -> MagicMock:
    row = MagicMock(spec=PrismaWorkloadAccessToken)
    row.id = "wt-1"
    row.token = token
    row.createdAt = _NOW
    row.expiresAt = expires_at or _NOW + timedelta(minutes=15)
    row.workloadIdentityId = "wi-1"
    row.scopes = scopes or [APIKeyPermission.READ_GRAPH]
    row.subject = "repo:acme/api:ref:refs/heads/main"
    row.subjectJti = None
    row.revokedAt = revoked_at
    row.WorkloadIdentity = _identity_row(status=identity_status)
    return row
