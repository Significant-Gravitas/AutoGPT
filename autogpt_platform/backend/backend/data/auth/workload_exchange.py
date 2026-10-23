"""Exchange a workload's OIDC token for a short-lived AutoGPT token (RFC 8693)."""

import logging
from typing import Any

from prisma.enums import APIKeyPermission, WorkloadIdentityStatus

from backend.data.user import is_user_banned

from .oidc import OIDCVerificationError, verify_oidc_token
from .workload_identity import (
    WorkloadIdentityInfo,
    WorkloadTokenInfo,
    get_workload_identity,
    mint_workload_token,
    record_workload_identity_use,
)
from .workload_trust import claims_match

logger = logging.getLogger(__name__)


class TokenExchangeError(Exception):
    """The exchange was refused; ``error`` is the RFC 6749 error code."""

    error = "invalid_request"


class InvalidSubjectTokenError(TokenExchangeError):
    error = "invalid_grant"


class InvalidScopeError(TokenExchangeError):
    error = "invalid_scope"


async def exchange_for_workload_token(
    identity_id: str, subject_token: str, requested_scopes: list[str] | None
) -> tuple[WorkloadTokenInfo, str]:
    """Mint a token for the identity if ``subject_token`` is an OIDC token it
    trusts. Returns the token's info and its plaintext."""
    identity = await get_workload_identity(identity_id)
    if identity is None or identity.status != WorkloadIdentityStatus.ACTIVE:
        raise InvalidSubjectTokenError("Unknown or inactive workload identity")
    if await is_user_banned(identity.user_id):
        raise InvalidSubjectTokenError("The workload identity's owner is banned")
    claims = await _trusted_claims(identity, subject_token)
    scopes = _granted_scopes(identity.permissions, requested_scopes)
    token, plaintext = await mint_workload_token(
        identity,
        subject=str(claims["sub"]),
        subject_jti=str(claims["jti"]) if claims.get("jti") else None,
        scopes=scopes,
    )
    await record_workload_identity_use(identity)
    logger.info(
        f"Issued workload token #{token.id} for identity #{identity.id} "
        f"to subject {claims['sub']!r}"
    )
    return token, plaintext


async def _trusted_claims(
    identity: WorkloadIdentityInfo, subject_token: str
) -> dict[str, Any]:
    try:
        claims = await verify_oidc_token(
            subject_token, issuer=identity.issuer, audience=identity.audience
        )
    except OIDCVerificationError as e:
        raise InvalidSubjectTokenError(str(e)) from e
    if not claims_match(identity.claim_conditions, claims):
        raise InvalidSubjectTokenError(
            "Token does not meet the workload identity's claim conditions"
        )
    return claims


def _granted_scopes(
    allowed: list[APIKeyPermission], requested: list[str] | None
) -> list[APIKeyPermission]:
    """The requested subset of the identity's permissions; all if none asked."""
    if not requested:
        return list(allowed)
    known = {p.value for p in APIKeyPermission}
    unknown = [s for s in requested if s not in known]
    if unknown:
        raise InvalidScopeError(f"Unknown scopes: {', '.join(unknown)}")
    wanted = list(dict.fromkeys(APIKeyPermission(s) for s in requested))
    missing = [p.value for p in wanted if p not in allowed]
    if missing:
        raise InvalidScopeError(
            f"Not granted to this workload identity: {', '.join(missing)}"
        )
    return wanted
