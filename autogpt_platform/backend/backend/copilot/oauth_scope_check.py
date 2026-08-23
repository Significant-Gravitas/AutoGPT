"""Tell the chat — and the model — when an OAuth connect granted too little.

A user connected GitHub from a copilot setup card, the OAuth round-trip
succeeded, and the token that came back carried **zero** scopes. Nothing
complained. The failure surfaced much later, at call time, as an opaque 403,
and cost three rounds of "try connecting again" before anyone worked out that
GitHub had silently reused a prior authorization and granted nothing new.

The requested-vs-granted diff already existed in the OAuth callback; it just
logged a warning nobody reads. This module is the missing half: it carries the
*provenance* of a copilot-initiated connect from the setup card to the
callback, and turns a shortfall into something both the user and the model
can act on.

Provenance is a short-TTL Redis record rather than a field on the OAuth state
token, because the copilot never mints the state token — ``connect_integration``
renders a card, the *frontend* calls ``/login``. Threading a session id through
would mean a new user-supplied query parameter on an auth endpoint plus a
frontend and OpenAPI change; a server-written record keyed by
``(user_id, provider)`` needs neither and cannot be influenced by the client.

The obvious hole in a ``(user, provider)`` key is a user with two connect cards
open for the same provider. Rather than guess which one a callback belongs to,
the key holds a bounded **list** of pending connects, and the callback drains
all of them and judges each *chat* against the union of everything its own
cards asked for. Both chats learn the truth about the connection they were
both waiting on; neither is told about scopes it never asked for.
"""

import json
import logging
import time
from typing import Any, Awaitable, cast

from pydantic import BaseModel, Field

from backend.data.redis_client import get_redis_async
from backend.integrations.oauth.scopes import (
    ScopeCoverageResult,
    evaluate_scope_coverage,
)
from backend.util.background import spawn_background_task
from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)

# Pending copilot-initiated connects, keyed by (user, provider). The TTL only
# has to cover "card rendered → user finishes the OAuth popup"; a connect the
# user comes back to an hour later simply isn't attributed to a chat.
_PENDING_PREFIX = "copilot:oauth_pending_connect:"
_PENDING_TTL_SECONDS = 30 * 60
_MAX_PENDING_PER_PROVIDER = 5

# Model-facing shortfall status, keyed by (user, session), one hash field per
# provider. Read once per turn by ``build_session_context``.
_STATUS_PREFIX = "copilot:oauth_scope_status:"
_STATUS_TTL_SECONDS = 24 * 60 * 60

# Claim marker so a replayed callback cannot post the same notice twice.
_NOTICE_CLAIM_PREFIX = "copilot:oauth_scope_notice:"
_NOTICE_CLAIM_TTL_SECONDS = 60 * 60

_GENERIC_REMEDIATION = (
    "Disconnect the app in the provider's account settings so it has to ask "
    "again, then reconnect and leave every permission on the consent screen "
    "checked."
)
_REMEDIATION_BY_PROVIDER: dict[str, str] = {
    "github": (
        "GitHub silently reuses a previous authorization instead of showing "
        "the consent screen again, so reconnecting on its own changes "
        "nothing. Revoke the app first at GitHub Settings → Applications → "
        "Authorized OAuth Apps, then reconnect."
    ),
}


def remediation_for(provider: str) -> str:
    return _REMEDIATION_BY_PROVIDER.get(provider.lower(), _GENERIC_REMEDIATION)


class PendingConnect(BaseModel):
    """One copilot setup card awaiting an OAuth round-trip."""

    session_id: str
    requested_scopes: list[str] = Field(default_factory=list)
    created_at: int


def _pending_key(user_id: str, provider: str) -> str:
    return f"{_PENDING_PREFIX}{user_id}:{provider.lower()}"


def status_key(user_id: str, session_id: str) -> str:
    return f"{_STATUS_PREFIX}{user_id}:{session_id}"


async def record_pending_connect(
    user_id: str, provider: str, session_id: str, requested_scopes: list[str]
) -> None:
    """Remember that *session_id* asked *user_id* to connect *provider*.

    Best-effort: a Redis hiccup here must not stop the setup card from
    rendering, it only means a later shortfall goes unattributed.
    """
    try:
        record = PendingConnect(
            session_id=session_id,
            requested_scopes=requested_scopes,
            created_at=int(time.time()),
        )
        key = _pending_key(user_id, provider)
        redis = await get_redis_async()
        async with redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, record.model_dump_json())
            # Keep the newest N: a user who keeps re-opening cards should not
            # be able to grow this unboundedly.
            pipe.ltrim(key, -_MAX_PENDING_PER_PROVIDER, -1)
            pipe.expire(key, _PENDING_TTL_SECONDS)
            await cast(Awaitable[list[Any]], pipe.execute())
    except Exception:
        logger.warning(
            "could not record pending %s connect for session=%s",
            provider,
            session_id[:12],
            exc_info=True,
        )


async def _drain_pending_connects(user_id: str, provider: str) -> list[PendingConnect]:
    """Atomically read and clear every pending connect for (user, provider)."""
    key = _pending_key(user_id, provider)
    redis = await get_redis_async()
    async with redis.pipeline(transaction=True) as pipe:
        pipe.lrange(key, 0, -1)
        pipe.delete(key)
        results = await cast(Awaitable[list[Any]], pipe.execute())

    records: list[PendingConnect] = []
    for raw in cast(list[str], results[0] or []):
        try:
            records.append(PendingConnect.model_validate_json(raw))
        except Exception:
            logger.warning("discarding unparseable pending connect record")
    return records


def schedule_scope_check(
    *,
    user_id: str,
    provider: str,
    granted_scopes: list[str],
    provider_reports_scopes: bool,
    username: str | None,
) -> None:
    """Fire-and-forget the post-connect scope reconciliation.

    Detached on purpose: the OAuth callback has already stored a working
    credential by the time this runs, and nothing here is worth failing that
    for.
    """
    spawn_background_task(
        _run_scope_check(
            user_id=user_id,
            provider=provider,
            granted_scopes=granted_scopes,
            provider_reports_scopes=provider_reports_scopes,
            username=username,
        ),
        name=f"oauth-scope-check-{provider}",
    )


async def _run_scope_check(
    *,
    user_id: str,
    provider: str,
    granted_scopes: list[str],
    provider_reports_scopes: bool,
    username: str | None,
) -> None:
    try:
        pending = await _drain_pending_connects(user_id, provider)
        if not pending:
            return

        for session_id, requested_scopes in _requested_scopes_by_session(
            pending
        ).items():
            result = evaluate_scope_coverage(
                requested_scopes,
                granted_scopes,
                provider_reports_scopes=provider_reports_scopes,
            )
            if not result.is_shortfall:
                # A good reconnect clears whatever the last bad one left
                # behind, so the model stops warning about a fixed problem.
                await _clear_status(user_id, session_id, provider)
                continue

            await _write_status(user_id, session_id, provider, result)
            await _post_chat_notice(
                user_id=user_id,
                session_id=session_id,
                provider=provider,
                result=result,
                username=username,
            )
    except Exception:
        logger.warning(
            "post-connect scope check failed for provider=%s; dropping",
            provider,
            exc_info=True,
        )


def _requested_scopes_by_session(
    pending: list[PendingConnect],
) -> dict[str, list[str]]:
    """Collapse one session's cards into the union of what they asked for.

    ``record_pending_connect`` runs on every ``connect_integration`` call, so
    a model that re-renders a widened card leaves a second record for the same
    session. The user saw both cards and both asked for something, so the
    honest thing to judge the grant against is everything that was on screen —
    not whichever card happened to be recorded first, which would let a grant
    that satisfies the narrow card bury the wider card's shortfall.
    """
    grouped: dict[str, list[str]] = {}
    for record in pending:
        grouped.setdefault(record.session_id, []).extend(record.requested_scopes)
    return grouped


# --------------------------------------------------------------------------- #
# Model-facing status (read per turn by ``build_session_context``)
# --------------------------------------------------------------------------- #


async def _write_status(
    user_id: str, session_id: str, provider: str, result: ScopeCoverageResult
) -> None:
    payload = json.dumps(
        {
            "coverage": result.coverage.value,
            "requested": result.requested,
            "granted": result.granted,
            "missing": result.missing,
        }
    )
    key = status_key(user_id, session_id)
    redis = await get_redis_async()
    async with redis.pipeline(transaction=True) as pipe:
        pipe.hset(key, provider.lower(), payload)
        pipe.expire(key, _STATUS_TTL_SECONDS)
        await cast(Awaitable[list[Any]], pipe.execute())


async def _clear_status(user_id: str, session_id: str, provider: str) -> None:
    redis = await get_redis_async()
    await cast(
        Awaitable[int], redis.hdel(status_key(user_id, session_id), provider.lower())
    )


async def scope_status_lines(user_id: str, session_id: str) -> list[str]:
    """Render any outstanding scope shortfalls for this session.

    Returned unconditionally — not behind the notice feature flag. The flag
    gates whether we *interrupt the user*; the model should always know that
    a connection it can see is narrower than it asked for, so it explains the
    gap instead of re-firing ``connect_integration`` into the same wall.

    Any failure yields no lines: this runs on the per-turn hot path.
    """
    try:
        redis = await get_redis_async()
        entries = cast(
            dict[str, str],
            await cast(
                Awaitable[dict[Any, Any]],
                redis.hgetall(status_key(user_id, session_id)),
            ),
        )
    except Exception:
        logger.debug("could not read oauth scope status", exc_info=True)
        return []

    lines: list[str] = []
    for provider, raw in sorted(entries.items()):
        try:
            data = json.loads(raw)
        except Exception:
            continue
        missing = ", ".join(data.get("missing") or []) or "(all requested)"
        granted = ", ".join(data.get("granted") or []) or "none"
        lines.append(
            f"credential_scope_shortfall: {provider} is connected but the "
            f"granted token is missing {missing} (granted: {granted}). "
            f"Do not silently retry connect_integration for {provider} — tell "
            f"the user what is missing and how to fix it: "
            f"{remediation_for(provider)}"
        )
    return lines


# --------------------------------------------------------------------------- #
# User-facing chat notice (feature-flagged, default off)
# --------------------------------------------------------------------------- #


async def _post_chat_notice(
    *,
    user_id: str,
    session_id: str,
    provider: str,
    result: ScopeCoverageResult,
    username: str | None,
) -> None:
    if not await is_feature_enabled(
        Flag.COPILOT_OAUTH_SCOPE_CHECK, user_id, default=False
    ):
        return

    # Local imports: ``model`` and ``session_waiter`` pull in the copilot
    # runtime, which this module is imported *by* (session context) and
    # *into* (the integrations router).
    from backend.copilot.model import get_chat_session_metadata
    from backend.copilot.sdk.session_waiter import run_copilot_turn_via_queue

    # Ownership re-check. The pending record is already namespaced by user id,
    # so this cannot normally fail — but a wake posts into a chat transcript,
    # and that is not a place to rely on "cannot normally".
    session = await get_chat_session_metadata(session_id, user_id)
    if session is None:
        return

    if not await _claim_notice(session_id, provider, result):
        return

    outcome, _ = await run_copilot_turn_via_queue(
        session_id=session_id,
        user_id=user_id,
        message=shortfall_message(provider=provider, result=result, username=username),
        # 0 = don't wait: an idle chat gets a fresh turn, a busy one gets the
        # message on its pending buffer.
        timeout=0,
        tool_call_id=f"oauthscope:{provider}",
        tool_name="oauth_scope_shortfall",
    )
    logger.info(
        "oauth scope shortfall notice enqueued on session=%s for provider=%s "
        "(missing=%s, outcome=%s)",
        session_id[:12],
        provider,
        result.missing,
        outcome,
    )


def shortfall_message(
    *, provider: str, result: ScopeCoverageResult, username: str | None
) -> str:
    """The system-framed prompt the chat wakes up to.

    Buffered messages carry no author field — everything on the pending
    buffer is presented to the model as the user — so the framing is the only
    thing stopping the model from answering this line as if the user typed
    it. Same convention as ``subsession_wake.wake_message``.
    """
    connected_as = f" as `{username}`" if username else ""
    missing = ", ".join(result.missing) or "the requested permissions"
    granted = ", ".join(result.granted) or "nothing"
    return (
        "[System notice, not the user speaking: an OAuth connection the user "
        "just completed came back with fewer permissions than were "
        "requested. Do not reply to this notice itself.]\n\n"
        f'<oauth_scope_shortfall provider="{provider}" '
        f'coverage="{result.coverage.value}" missing="{missing}" />\n\n'
        f"The {provider} account is connected{connected_as}, but the token "
        f"only carries {granted}, so {missing} was not granted. Tell the user "
        "this now, in your own voice and in one short paragraph: what "
        "connected, exactly which permission is missing, and this fix — "
        f"{remediation_for(provider)} Do not re-run the connect step for them "
        "until they have done that; it would hand back the same token."
    )


async def _claim_notice(
    session_id: str, provider: str, result: ScopeCoverageResult
) -> bool:
    """Claim the right to post this exact shortfall once.

    Keyed on the missing set as well as the session, so a *different*
    shortfall after a partial re-auth is still reported.
    """
    redis = await get_redis_async()
    key = (
        f"{_NOTICE_CLAIM_PREFIX}{session_id}:{provider.lower()}:"
        f"{','.join(sorted(result.missing))}"
    )
    return bool(await redis.set(key, "1", nx=True, ex=_NOTICE_CLAIM_TTL_SECONDS))
