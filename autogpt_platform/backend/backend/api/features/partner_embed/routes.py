"""Restricted API facade for embedded partner clients."""

from typing import Annotated

from autogpt_libs.auth import RequestContext, requires_frontend_service
from fastapi import APIRouter, HTTPException, Query, Response, Security, status
from fastapi.responses import StreamingResponse

from backend.api.features.chat.routes import StreamChatRequest, stream_chat_post
from backend.api.features.partner_embed.auth import EmbedPrincipal, require_embed_scope
from backend.api.features.partner_embed.llm_route import resolve_embed_llm_route
from backend.api.features.partner_embed.models import (
    CreateEmbedSessionResponse,
    EmbedArtifact,
    EmbedSessionDetailResponse,
    EmbedSessionSummary,
    EmbedTurnRequest,
    ListEmbedArtifactsResponse,
    ListEmbedSessionsResponse,
    ProvisionPartnerIdentityRequest,
    ProvisionPartnerIdentityResponse,
    UpdateEmbedSessionTitleRequest,
    UpdateEmbedSessionTitleResponse,
)
from backend.api.features.partner_embed.provisioning import provision_partner_identity
from backend.api.features.workspace.routes import create_file_download_response
from backend.copilot.db import get_chat_messages_paginated
from backend.copilot.model import (
    ChatSessionInfo,
    create_chat_session,
    get_chat_session_metadata,
    get_user_sessions,
    update_session_title,
)
from backend.copilot.service import (
    _fallback_title_from_message,
    strip_injected_context_for_display,
)
from backend.data.workspace import get_or_create_workspace
from backend.util.workspace import WorkspaceManager

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
    llm_auth_provider, llm_credential_id = await resolve_embed_llm_route(
        principal.user_id
    )
    session = await create_chat_session(
        principal.user_id,
        dry_run=False,
        organization_id=principal.organization_id,
        team_id=principal.team_id,
        source_platform=principal.partner_id,
        external_account_id=principal.external_account_id,
        external_capabilities=principal.capabilities,
        llm_auth_provider=llm_auth_provider,
        llm_credential_id=llm_credential_id,
    )
    return CreateEmbedSessionResponse(
        id=session.session_id,
        created_at=session.started_at.isoformat(),
    )


@router.get("/sessions", summary="List embedded chat sessions")
async def list_embed_sessions(
    principal: EmbedChatPrincipal,
) -> ListEmbedSessionsResponse:
    sessions, _ = await get_user_sessions(
        principal.user_id,
        limit=200,
        organization_id=principal.organization_id,
        pinned_first=False,
    )
    visible = [
        session
        for session in sessions
        if _session_matches_principal(session, principal)
    ]
    return ListEmbedSessionsResponse(
        sessions=[
            EmbedSessionSummary(
                id=session.session_id,
                title=session.title,
                created_at=session.started_at.isoformat(),
                updated_at=session.updated_at.isoformat(),
                chat_status=session.chat_status,
            )
            for session in visible
        ]
    )


@router.get(
    "/sessions/{session_id}",
    summary="Get an embedded chat session",
)
async def get_embed_session(
    session_id: str,
    principal: EmbedChatPrincipal,
    limit: int = Query(default=100, ge=1, le=200),
    before_sequence: int | None = Query(default=None, ge=0),
) -> EmbedSessionDetailResponse:
    session = await _require_embed_session(session_id, principal)
    page = await get_chat_messages_paginated(
        session_id,
        limit,
        before_sequence,
        user_id=principal.user_id,
        organization_id=principal.organization_id,
    )
    if page is None:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = []
    for message in page.messages:
        public_message = message.model_dump()
        if message.role == "user" and isinstance(message.content, str):
            public_message["content"] = strip_injected_context_for_display(
                message.content
            )
        messages.append(public_message)
    return EmbedSessionDetailResponse(
        id=session.session_id,
        title=session.title,
        created_at=session.started_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
        chat_status=session.chat_status,
        messages=messages,
        has_more_messages=page.has_more,
        oldest_sequence=page.oldest_sequence,
        capabilities=session.metadata.external_capabilities,
    )


@router.patch(
    "/sessions/{session_id}/title",
    summary="Set a fallback title for an embedded chat session",
)
async def update_embed_session_title(
    session_id: str,
    request: UpdateEmbedSessionTitleRequest,
    principal: EmbedChatPrincipal,
) -> UpdateEmbedSessionTitleResponse:
    session = await _require_embed_session(session_id, principal)
    if session.title:
        return UpdateEmbedSessionTitleResponse(title=session.title)
    title = _fallback_title_from_message(request.message)
    if await update_session_title(
        session_id,
        principal.user_id,
        title,
        only_if_empty=True,
    ):
        return UpdateEmbedSessionTitleResponse(title=title)
    latest = await _require_embed_session(session_id, principal)
    if latest.title:
        return UpdateEmbedSessionTitleResponse(title=latest.title)
    raise HTTPException(status_code=503, detail="Unable to title session")


@router.get(
    "/sessions/{session_id}/artifacts",
    summary="List artifacts generated in an embedded chat session",
)
async def list_embed_artifacts(
    session_id: str,
    principal: EmbedChatPrincipal,
) -> ListEmbedArtifactsResponse:
    await _require_embed_session(session_id, principal)
    _require_capability(principal, "documents.read")
    manager = await _workspace_manager(principal.user_id, session_id)
    files = await manager.list_files(limit=200)
    return ListEmbedArtifactsResponse(
        artifacts=[
            EmbedArtifact(
                id=file.id,
                name=file.name,
                path=file.path,
                mime_type=file.mime_type,
                size_bytes=file.size_bytes,
                created_at=file.created_at.isoformat(),
            )
            for file in files
        ]
    )


@router.get(
    "/sessions/{session_id}/artifacts/{file_id}/download",
    summary="Download an artifact generated in an embedded chat session",
)
async def download_embed_artifact(
    session_id: str,
    file_id: str,
    principal: EmbedChatPrincipal,
) -> Response:
    await _require_embed_session(session_id, principal)
    _require_capability(principal, "documents.read")
    manager = await _workspace_manager(principal.user_id, session_id)
    file = await manager.get_file_info(file_id)
    expected_prefix = f"/sessions/{session_id}/"
    if file is None or not file.path.startswith(expected_prefix):
        raise HTTPException(status_code=404, detail="Artifact not found")
    return await create_file_download_response(file)


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
    if session is None or not _session_matches_principal(session, principal):
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _session_matches_principal(
    session: ChatSessionInfo,
    principal: EmbedPrincipal,
) -> bool:
    return (
        session.organization_id == principal.organization_id
        and session.team_id == principal.team_id
        and session.metadata.source_platform == principal.partner_id
        and session.metadata.external_account_id == principal.external_account_id
        and session.metadata.external_capabilities == principal.capabilities
    )


def _require_capability(principal: EmbedPrincipal, capability: str) -> None:
    if capability not in principal.capabilities:
        raise HTTPException(status_code=403, detail=f"Missing capability: {capability}")


async def _workspace_manager(user_id: str, session_id: str) -> WorkspaceManager:
    workspace = await get_or_create_workspace(user_id)
    return WorkspaceManager(user_id, workspace.id, session_id)
