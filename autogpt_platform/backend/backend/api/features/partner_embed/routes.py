"""Restricted API facade for embedded partner clients."""

from typing import Annotated

from autogpt_libs.auth import RequestContext, requires_frontend_service
from fastapi import APIRouter, HTTPException, Security, status
from fastapi.responses import StreamingResponse

from backend.api.features.chat.routes import StreamChatRequest, stream_chat_post
from backend.api.features.partner_embed.auth import EmbedPrincipal, require_embed_scope
from backend.api.features.partner_embed.models import (
    CreateEmbedSessionResponse,
    EmbedTurnRequest,
    ProvisionPartnerIdentityRequest,
    ProvisionPartnerIdentityResponse,
)
from backend.api.features.partner_embed.provisioning import provision_partner_identity
from backend.copilot.model import (
    ChatSessionInfo,
    create_chat_session,
    get_chat_session_metadata,
)

router = APIRouter(prefix="/api/embed/v1", tags=["partner-embed"])

requires_partner_provisioning_service = requires_frontend_service(
    "partner-embed:provision"
)
requires_embed_chat = require_embed_scope("embed:chat")
EmbedChatPrincipal = Annotated[EmbedPrincipal, Security(requires_embed_chat)]


@router.post(
    "/provision",
    dependencies=[Security(requires_partner_provisioning_service)],
    summary="Provision a verified partner identity",
)
async def provision_partner(
    request: ProvisionPartnerIdentityRequest,
) -> ProvisionPartnerIdentityResponse:
    return await provision_partner_identity(request)


@router.post(
    "/sessions",
    status_code=status.HTTP_201_CREATED,
    summary="Create an embedded chat session",
)
async def create_embed_session(
    principal: EmbedChatPrincipal,
) -> CreateEmbedSessionResponse:
    session = await create_chat_session(
        principal.user_id,
        dry_run=False,
        organization_id=principal.organization_id,
        team_id=principal.team_id,
        source_platform=principal.partner_id,
    )
    return CreateEmbedSessionResponse(
        id=session.session_id,
        created_at=session.started_at.isoformat(),
    )


@router.post(
    "/sessions/{session_id}/stream",
    summary="Start an embedded chat turn",
)
async def stream_embed_turn(
    session_id: str,
    request: EmbedTurnRequest,
    principal: EmbedChatPrincipal,
) -> StreamingResponse:
    await _require_embed_session(session_id, principal)
    context = RequestContext(
        user_id=principal.user_id,
        org_id=principal.organization_id,
        team_id=principal.team_id,
        is_org_owner=False,
        is_org_admin=False,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )
    return await stream_chat_post(
        session_id,
        StreamChatRequest(
            message=request.message,
            message_id=request.message_id,
        ),
        user_id=principal.user_id,
        ctx=context,
    )


async def _require_embed_session(
    session_id: str,
    principal: EmbedPrincipal,
) -> ChatSessionInfo:
    session = await get_chat_session_metadata(session_id, principal.user_id)
    if (
        session is None
        or session.organization_id != principal.organization_id
        or session.metadata.source_platform != principal.partner_id
    ):
        raise HTTPException(status_code=404, detail="Session not found")
    return session
