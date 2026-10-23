"""Exchanging a workload's OIDC token: who gets a token, and with which scopes."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from prisma.enums import APIKeyPermission, WorkloadIdentityStatus

from backend.data.auth import workload_exchange
from backend.data.auth.oidc import OIDCVerificationError
from backend.data.auth.workload_exchange import (
    InvalidScopeError,
    InvalidSubjectTokenError,
    exchange_for_workload_token,
)
from backend.data.auth.workload_identity import WorkloadIdentityInfo, WorkloadTokenInfo
from backend.data.auth.workload_trust import ClaimCondition

CLAIMS = {
    "iss": "https://token.actions.githubusercontent.com",
    "sub": "repo:acme/api:ref:refs/heads/main",
    "repository": "acme/api",
    "jti": "jti-1",
}


@pytest.fixture
def deps(mocker):
    identity = _identity()
    mocks = {
        "get_workload_identity": AsyncMock(return_value=identity),
        "is_user_banned": AsyncMock(return_value=False),
        "verify_oidc_token": AsyncMock(return_value=CLAIMS),
        "mint_workload_token": AsyncMock(
            side_effect=lambda identity, **kw: (_token_info(kw["scopes"]), "agpt_wt_x")
        ),
        "record_workload_identity_use": AsyncMock(),
    }
    for name, mock in mocks.items():
        mocker.patch.object(workload_exchange, name, mock)
    return mocks


@pytest.mark.asyncio
async def test_a_trusted_token_gets_all_the_identitys_permissions(deps):
    token, plaintext = await exchange_for_workload_token("wi-1", "oidc", None)

    assert plaintext == "agpt_wt_x"
    deps["verify_oidc_token"].assert_awaited_once_with(
        "oidc",
        issuer="https://token.actions.githubusercontent.com",
        audience="https://backend.agpt.co",
    )
    mint = deps["mint_workload_token"].await_args.kwargs
    assert mint["subject"] == "repo:acme/api:ref:refs/heads/main"
    assert mint["subject_jti"] == "jti-1"
    assert set(mint["scopes"]) == {
        APIKeyPermission.EXECUTE_GRAPH,
        APIKeyPermission.READ_GRAPH,
    }
    deps["record_workload_identity_use"].assert_awaited_once()


@pytest.mark.asyncio
async def test_a_requested_subset_of_scopes_is_all_that_is_granted(deps):
    await exchange_for_workload_token("wi-1", "oidc", ["READ_GRAPH"])

    assert deps["mint_workload_token"].await_args.kwargs["scopes"] == [
        APIKeyPermission.READ_GRAPH
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("scopes", [["WRITE_GRAPH"], ["NOT_A_SCOPE"]])
async def test_scopes_beyond_the_identity_are_refused(deps, scopes):
    with pytest.raises(InvalidScopeError) as exc:
        await exchange_for_workload_token("wi-1", "oidc", scopes)

    assert exc.value.error == "invalid_scope"
    deps["mint_workload_token"].assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setup",
    [
        lambda d: setattr(d["get_workload_identity"], "return_value", None),
        lambda d: setattr(
            d["get_workload_identity"],
            "return_value",
            _identity(status=WorkloadIdentityStatus.DISABLED),
        ),
        lambda d: setattr(
            d["get_workload_identity"],
            "return_value",
            _identity(status=WorkloadIdentityStatus.REVOKED),
        ),
        lambda d: setattr(d["is_user_banned"], "return_value", True),
        lambda d: setattr(
            d["verify_oidc_token"], "side_effect", OIDCVerificationError("bad sig")
        ),
        lambda d: setattr(
            d["verify_oidc_token"],
            "return_value",
            {**CLAIMS, "repository": "mallory/api"},
        ),
    ],
    ids=[
        "unknown identity",
        "disabled identity",
        "revoked identity",
        "banned owner",
        "unverifiable token",
        "claims not matching",
    ],
)
async def test_an_untrusted_exchange_mints_nothing(deps, setup):
    setup(deps)

    with pytest.raises(InvalidSubjectTokenError) as exc:
        await exchange_for_workload_token("wi-1", "oidc", None)

    assert exc.value.error == "invalid_grant"
    deps["mint_workload_token"].assert_not_awaited()


def _identity(status=WorkloadIdentityStatus.ACTIVE) -> WorkloadIdentityInfo:
    return WorkloadIdentityInfo(
        id="wi-1",
        name="deploy",
        issuer="https://token.actions.githubusercontent.com",
        audience="https://backend.agpt.co",
        claim_conditions=[ClaimCondition(claim="repository", pattern="acme/api")],
        permissions=[APIKeyPermission.EXECUTE_GRAPH, APIKeyPermission.READ_GRAPH],
        token_ttl_seconds=900,
        status=status,
        created_at=datetime.now(timezone.utc),
        user_id="user-1",
    )


def _token_info(scopes: list[APIKeyPermission]) -> WorkloadTokenInfo:
    return WorkloadTokenInfo(
        id="wt-1",
        type="workload",
        user_id="user-1",
        scopes=scopes,
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc),
        workload_identity_id="wi-1",
        subject=CLAIMS["sub"],
    )
