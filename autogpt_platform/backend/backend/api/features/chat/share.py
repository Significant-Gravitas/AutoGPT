"""HTTP routes for sharing chat sessions.

Three groups of routes:

- **Owner-only** — list linked-execution candidates, enable share with
  opt-ins, disable share.  Mirrors the ``/graphs/.../share`` shape on
  :mod:`backend.api.features.v1` for execution sharing.
- **Public-by-token** — the public viewer reads through these
  unauthenticated routes; the share token is the bearer credential.
  Same enumeration defenses as execution sharing (strict UUID path
  validation, allowlist lookup, uniform 404).
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from autogpt_libs import auth
from fastapi import APIRouter, Body, HTTPException, Path, Response, Security
from pydantic import BaseModel
from starlette.status import HTTP_204_NO_CONTENT

from backend.api.features.workspace.routes import create_file_download_response
from backend.copilot.model import get_chat_session_metadata
from backend.copilot.sharing import db as share_db
from backend.copilot.sharing.models import SharedChatMessagesPage, SharedChatSession
from backend.data.db_accessors import (
    LiveResourceAccessRevoked,
    require_exact_chat_session_scope,
)
from backend.data.sharing.tokens import SHARE_TOKEN_PATTERN
from backend.data.tenancy import live_resource_permission_barrier
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

_view_resource_context = auth.requires_resource_permission(
    auth.OrgAction.VIEW_RESOURCES,
    auth.TeamAction.VIEW_AGENTS,
)
_create_resource_context = auth.requires_resource_permission(
    auth.OrgAction.CREATE_RESOURCES,
    auth.TeamAction.CREATE_AGENTS,
)


@asynccontextmanager
async def _live_chat_share_action(
    session_id: str,
    user_id: str,
    ctx: auth.RequestContext,
    org_action: auth.OrgAction,
    team_action: auth.TeamAction,
) -> AsyncIterator[tuple[str | None, str | None]]:
    session = await get_chat_session_metadata(session_id, user_id)
    if (
        session is None
        or session.organization_id != ctx.org_id
        or (ctx.team_id is not None and session.team_id != ctx.team_id)
    ):
        raise HTTPException(status_code=404, detail="Chat session not found")
    try:
        async with live_resource_permission_barrier(
            user_id,
            session.organization_id,
            session.team_id,
            org_action,
            team_action,
        ) as allowed:
            if not allowed:
                raise LiveResourceAccessRevoked("workspace_access_revoked")
            await require_exact_chat_session_scope(
                session_id,
                user_id,
                session.organization_id,
                session.team_id,
            )
            yield session.organization_id, session.team_id
    except LiveResourceAccessRevoked as exc:
        raise HTTPException(status_code=404, detail="Chat session not found") from exc


# --------------------------------------------------------------------------
# Owner-only routes — mounted at the chat router's prefix (``/api/chat``).
# --------------------------------------------------------------------------

owner_router = APIRouter(tags=["chat", "share"])


class ChatShareStateResponse(BaseModel):
    """Surfaces the chat's current share state so the modal can open
    in the right mode (share-vs-revoke) without an extra round-trip.

    The three counts power the consent-disclosure block above the
    toggle so the owner sees exactly what they're about to expose.
    They reflect live state — sharing is live (not snapshot-at-enable),
    so these numbers are also a lower bound for what subsequent
    viewers will see if more messages / runs / files land before
    revocation.
    """

    is_shared: bool = False
    share_token: str | None = None
    auto_share_executions: bool = False
    message_count: int = 0
    linked_run_count: int = 0
    file_count: int = 0


class EnableShareRequest(BaseModel):
    """Whether to auto-link every ``run_agent`` execution in this chat.

    When true, the backend creates ``ChatLinkedShare`` rows for every
    existing agent run AND auto-links any new runs that land while the
    share is active.  When false, only the chat messages are shared
    (cards in the viewer render in their "execution not shared" state).
    """

    auto_share_executions: bool = False


class ShareResponse(BaseModel):
    share_url: str
    share_token: str


@owner_router.get("/sessions/{session_id}/share/state")
async def get_chat_share_state(
    session_id: Annotated[str, Path],
    user_id: Annotated[str, Security(auth.get_user_id)],
    ctx: Annotated[auth.RequestContext, Security(_view_resource_context)],
) -> ChatShareStateResponse:
    """Return the chat's current share state.

    The share modal calls this on open so it can render in the right
    mode (share-vs-revoke) and pre-select the auto-share toggle.
    """
    async with _live_chat_share_action(
        session_id,
        user_id,
        ctx,
        auth.OrgAction.VIEW_RESOURCES,
        auth.TeamAction.VIEW_AGENTS,
    ) as (organization_id, team_id):
        state = await share_db.get_chat_share_state(
            session_id=session_id,
            user_id=user_id,
            organization_id=organization_id,
            team_id=team_id,
        )
    return ChatShareStateResponse(
        is_shared=state.is_shared,
        share_token=state.share_token,
        auto_share_executions=state.auto_share_executions,
        message_count=state.message_count,
        linked_run_count=state.linked_run_count,
        file_count=state.file_count,
    )


@owner_router.post(
    "/sessions/{session_id}/share",
    responses={
        403: {"description": "Chat sharing is not enabled for this user"},
        404: {"description": "Chat session not found for user"},
    },
)
async def enable_chat_sharing(
    session_id: Annotated[str, Path],
    user_id: Annotated[str, Security(auth.get_user_id)],
    ctx: Annotated[auth.RequestContext, Security(_create_resource_context)],
    body: EnableShareRequest = Body(default_factory=EnableShareRequest),
) -> ShareResponse:
    """Enable sharing for a chat session.

    Flag-gated: refuses with 403 when ``chat-sharing`` is off so a stale
    frontend cannot enable shares post-rollback.
    """
    if not await is_feature_enabled(Flag.CHAT_SHARING, user_id):
        raise HTTPException(status_code=403, detail="Chat sharing is not enabled")

    base_url = settings.config.frontend_base_url
    if not base_url:
        # Fail fast rather than handing the user a localhost URL that
        # only works on the backend host.  This catches deployment
        # misconfigurations at share-enable time instead of silently
        # shipping broken share URLs to end users.
        logger.error("frontend_base_url is not configured; refusing to enable share")
        raise HTTPException(
            status_code=500, detail="Sharing is not configured on this deployment"
        )

    try:
        async with _live_chat_share_action(
            session_id,
            user_id,
            ctx,
            auth.OrgAction.CREATE_RESOURCES,
            auth.TeamAction.CREATE_AGENTS,
        ) as (organization_id, team_id):
            share_token = await share_db.enable_chat_session_share(
                session_id=session_id,
                user_id=user_id,
                auto_share_executions=body.auto_share_executions,
                organization_id=organization_id,
                team_id=team_id,
            )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return ShareResponse(
        share_url=f"{base_url}/share/chat/{share_token}",
        share_token=share_token,
    )


@owner_router.delete(
    "/sessions/{session_id}/share",
    status_code=HTTP_204_NO_CONTENT,
)
async def disable_chat_sharing(
    session_id: Annotated[str, Path],
    user_id: Annotated[str, Security(auth.get_user_id)],
    ctx: Annotated[auth.RequestContext, Security(_create_resource_context)],
) -> None:
    """Revoke sharing for a chat session.

    Cascade-revokes linked executions whose share originated here
    (``sharedVia == CHAT_LINK``).  User-initiated execution shares
    survive — see :func:`backend.copilot.sharing.db.disable_chat_session_share`.
    """
    try:
        async with _live_chat_share_action(
            session_id,
            user_id,
            ctx,
            auth.OrgAction.CREATE_RESOURCES,
            auth.TeamAction.CREATE_AGENTS,
        ) as (organization_id, team_id):
            await share_db.disable_chat_session_share(
                session_id=session_id,
                user_id=user_id,
                organization_id=organization_id,
                team_id=team_id,
            )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# --------------------------------------------------------------------------
# Public routes — mounted at ``/api/public/shared/chats``.
# Stays on even when ``chat-sharing`` flag is off so revoked-then-fixed
# rollbacks don't break already-shared URLs mid-flight.
# --------------------------------------------------------------------------

public_router = APIRouter(tags=["chat", "share", "public"])


@public_router.get(
    "/{share_token}",
    responses={404: {"description": "Shared chat not found"}},
)
async def get_shared_chat(
    share_token: Annotated[str, Path(pattern=SHARE_TOKEN_PATTERN)],
) -> SharedChatSession:
    session = await share_db.get_chat_session_by_share_token(share_token)
    if not session:
        raise HTTPException(status_code=404, detail="Shared chat not found")
    return session


@public_router.get(
    "/{share_token}/messages",
    responses={
        400: {"description": "Invalid limit (must be 1..200)"},
        404: {"description": "Shared chat not found"},
    },
)
async def get_shared_chat_messages(
    share_token: Annotated[str, Path(pattern=SHARE_TOKEN_PATTERN)],
    limit: int = 50,
    before_sequence: int | None = None,
) -> SharedChatMessagesPage:
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=400, detail="limit must be in [1, 200]")
    page = await share_db.get_shared_chat_messages_paginated(
        share_token, limit=limit, before_sequence=before_sequence
    )
    if page is None:
        raise HTTPException(status_code=404, detail="Shared chat not found")
    return page


@public_router.get(
    "/{share_token}/files/{file_id}/download",
    summary="Download a file from a shared chat",
    operation_id="download_shared_chat_file",
    responses={
        404: {
            "description": (
                "Uniform 404 for every failure mode (unknown token, wrong "
                "file id, file no longer present) — prevents enumeration."
            )
        },
    },
)
async def download_shared_chat_file(
    share_token: Annotated[str, Path(pattern=SHARE_TOKEN_PATTERN)],
    file_id: Annotated[str, Path(pattern=SHARE_TOKEN_PATTERN)],
) -> Response:
    """Download a workspace file allowlisted by a shared chat (no auth).

    Returns uniform 404 for every failure mode to prevent enumeration —
    indistinguishable from "unknown token" or "wrong file id".
    """
    async with share_db.shared_chat_file_access(share_token, file_id) as file:
        if file is None:
            raise HTTPException(status_code=404, detail="Not found")
        return await create_file_download_response(file, inline=True)
