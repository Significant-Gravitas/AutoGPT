"""REL-001 JWT revocation write path.

The frontend mints short-lived JWTs (5m, auth.ts expirationTime) bound to
a Better Auth session via ``jti`` (per-token) + ``sid`` (per-session).
On logout or password-reset the session row is deleted, but the already-
issued JWT remains valid for up to 5m by signature alone. The revoke
endpoint writes ``revoked:jti:{jti}`` and ``revoked:sid:{sid}`` to the
shared Redis cluster (TTL 300s) so ``jwt_utils._is_jti_revoked`` rejects
any replay within the window. Fail-open: if Redis is down the request
still succeeds; exposure is bounded by the 5m expiry.

TTL choice: REVOKED_JTI_TTL_SECONDS = 300, REVOKED_SID_TTL_SECONDS = 300
(see jwt_utils.py). jti covers one token's lifetime. sid covers all
tokens from the session for the same window — no new tokens can be
minted after the DB session row is gone, so 300s is sufficient. Bump
sid to 86400 if clock-skew paranoia warrants, still cheap (one key per
logout).

Session coherence: ``better-auth.session_data`` (cookieCache maxAge 5m)
is a signed snapshot that survives session-row deletion. The hard gate
is the backend JWT denylist + DB session check (getServerSession /
getToken). Edge middleware treats cookieCache as a hint: non-admin
roles are returned quickly, admin roles fall through to a DB-backed
fetch so a revoked admin cookie cannot keep privilege for 5m.
"""

import logging
from typing import Any

from fastapi import APIRouter, Security, status

from autogpt_libs.auth.jwt_utils import get_jwt_payload, revoke_token_payload

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/revoke",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke the current JWT (jti) and session (sid)",
    operation_id="revokeCurrentToken",
    responses={
        401: {"description": "Missing or invalid Authorization bearer token"},
        204: {"description": "Revocation markers written (or fail-open if Redis down)"},
    },
)
async def revoke_current_token(
    jwt_payload: dict[str, Any] = Security(get_jwt_payload),
) -> None:
    # jwt_payload is already validated by get_jwt_payload; jti/sid came from
    # auth.ts definePayload and may be absent on legacy tokens — revoke what
    # we have. Legacy Supabase tokens (no jti/sid) skip revocation and rely
    # on 5m expiry; they are already bounded.
    jti: str | None = jwt_payload.get("jti")
    sid: str | None = jwt_payload.get("sid")
    if not jti and not sid:
        # No revocable claims — legacy token. Nothing to write; 5m expiry
        # bounds replay. Log for visibility.
        logger.info("revoke: legacy token without jti/sid, no marker written")
        return
    ok = revoke_token_payload(jwt_payload)
    if not ok:
        # Redis failure — fail-open, JWT still expires in ≤5m
        logger.warning(
            "revoke: Redis write failed for jti=%s sid=%s (fail-open, bounded 5m)",
            jti,
            sid,
        )
    else:
        logger.info("revoke: wrote markers for jti=%s sid=%s", jti, sid)
