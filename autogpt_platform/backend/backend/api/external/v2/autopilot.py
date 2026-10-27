"""
V2 External API - AutoPilot Endpoints

Read-only access to the caller's AutoPilot chat sessions, default and expert,
with their live status. No message bodies: this is a status feed.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Security
from prisma.enums import APIKeyPermission
from pydantic import BaseModel, Field

from backend.copilot import stream_registry
from backend.copilot.model import ChatSessionInfo, get_user_sessions
from backend.data.db_accessors import experts_db
from backend.data.human_review import chat_sessions_awaiting_review

from .pagination import Page, PageRequest
from .tenancy import TenantContext, require_permission

logger = logging.getLogger(__name__)

autopilot_router = APIRouter(tags=["autopilot"])

MAX_SESSIONS_PAGE_SIZE = 50
DEFAULT_SESSIONS_PAGE_SIZE = 10


class AutopilotSession(BaseModel):
    """One AutoPilot chat session and what it is doing right now."""

    id: str
    title: str | None = Field(description="`null` until the chat is titled")
    status: str = Field(
        description=(
            "Lifecycle state: `idle`, `queued` or `running`. Open set; treat an "
            "unknown value as `idle`."
        )
    )
    is_processing: bool = Field(description="A turn is streaming right now")
    needs_user: bool = Field(
        description="The chat is waiting on the user to approve or reject an action"
    )
    expert_id: str | None = Field(
        description="The expert this chat is with; `null` for the default AutoPilot"
    )
    expert_name: str | None = Field(
        description="The expert's display name, when it can be resolved"
    )
    source_platform: str | None = Field(
        description="Where the chat started, e.g. `discord`; `null` for the web app"
    )
    is_pinned: bool
    created_at: datetime
    updated_at: datetime


def session_page_request(
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=MAX_SESSIONS_PAGE_SIZE,
            description=f"Items per page (max {MAX_SESSIONS_PAGE_SIZE})",
        ),
    ] = DEFAULT_SESSIONS_PAGE_SIZE,
    cursor: Annotated[
        str | None,
        Query(description="`next_cursor` from the previous response"),
    ] = None,
) -> PageRequest:
    return PageRequest(limit=limit, cursor=cursor)


@autopilot_router.get(
    path="/sessions",
    summary="List AutoPilot sessions",
    operation_id="listAutopilotSessions",
)
async def list_autopilot_sessions(
    expert_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=128,
        description="Only the sessions with this expert",
    ),
    page: PageRequest = Depends(session_page_request),
    auth: TenantContext = Security(require_permission(APIKeyPermission.READ_AUTOPILOT)),
) -> Page[AutopilotSession]:
    """List the caller's AutoPilot sessions, most recently updated first.

    Covers default and expert sessions. `is_processing` and `needs_user` are
    live: poll this endpoint to follow them.
    """
    sessions, total_count = await get_user_sessions(
        auth.user_id,
        page.limit,
        (page.page - 1) * page.limit,
        organization_id=auth.organization_id,
        expert_id=expert_id,
        pinned_first=False,
    )
    session_ids = [s.session_id for s in sessions]

    running, waiting, expert_names = await asyncio.gather(
        stream_registry.running_session_ids(session_ids),
        _awaiting_review(auth.user_id, session_ids),
        _expert_names(auth.user_id, sessions),
    )

    return page.paged(
        [
            AutopilotSession(
                id=s.session_id,
                title=s.title,
                status=s.chat_status,
                is_processing=s.session_id in running,
                needs_user=s.session_id in waiting,
                expert_id=s.expert_id,
                expert_name=expert_names.get(s.expert_id) if s.expert_id else None,
                source_platform=s.metadata.source_platform,
                is_pinned=s.is_pinned,
                created_at=_utc(s.started_at),
                updated_at=_utc(s.updated_at),
            )
            for s in sessions
        ],
        total_count,
    )


async def _awaiting_review(user_id: str, session_ids: list[str]) -> set[str]:
    try:
        return await chat_sessions_awaiting_review(user_id, session_ids)
    except Exception:
        logger.warning("Failed to read pending reviews for AutoPilot sessions")
        return set()


async def _expert_names(
    user_id: str, sessions: list[ChatSessionInfo]
) -> dict[str, str]:
    if not any(s.expert_id for s in sessions):
        return {}
    try:
        identities = await experts_db().list_expert_identities(user_id)
    except Exception:
        logger.warning("Failed to resolve expert names for AutoPilot sessions")
        return {}
    return {e.id: e.name for e in identities}


def _utc(value: datetime) -> datetime:
    """Serialized with a `Z` suffix whatever offset the row came back with."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
