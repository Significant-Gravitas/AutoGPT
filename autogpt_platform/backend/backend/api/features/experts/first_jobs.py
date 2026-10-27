"""Each hired expert's first job, for the copilot home right after onboarding.

A hire does not copy its template's day-one rows, so the job comes from the
source template's first row. The status beside it is read from what the
Team page and the chat already know: whether setup is still installing or
missing a connection, whether the expert's latest chat is running, whether
its intake card still waits on the user, and whether a chat exists at all.
"""

import asyncio
from collections import defaultdict

import prisma.models
from pydantic import BaseModel

from backend.api.features.experts import experts_db, onboarding
from backend.api.features.experts.models import (
    Expert,
    ExpertDayOneItem,
    ExpertFirstJob,
    ExpertFirstJobsResponse,
    ExpertFirstJobStatus,
    ExpertSetupItem,
    decode_day_one,
)
from backend.api.features.experts.onboarding import OnboardingMessage
from backend.api.features.experts.setup import list_setup_items
from backend.copilot.db import exclude_dream_sessions_sql
from backend.copilot.model import CHAT_STATUS_QUEUED, CHAT_STATUS_RUNNING
from backend.data.db import query_raw_with_schema

_BUSY_CHAT_STATUSES = {CHAT_STATUS_RUNNING, CHAT_STATUS_QUEUED}
_CONNECTION_RESOLUTIONS = {"connect", "allow"}


class LatestExpertSession(BaseModel):
    expert_id: str
    session_id: str
    chat_status: str


class ExpertOnboardingMessage(OnboardingMessage):
    expert_id: str


async def list_first_jobs(user_id: str) -> ExpertFirstJobsResponse:
    """One card per live hire, oldest first."""
    experts = [
        expert
        for expert in await experts_db.list_experts(user_id, with_metrics=False)
        if not expert.is_archived and not expert.is_template
    ]
    if not experts:
        return ExpertFirstJobsResponse(items=[])
    expert_ids = [expert.id for expert in experts]
    template_ids = sorted(
        {e.source_template_id for e in experts if e.source_template_id}
    )
    day_ones, setup_items, sessions, onboardings = await asyncio.gather(
        _template_day_ones(template_ids),
        list_setup_items(user_id),
        _latest_sessions(user_id, expert_ids),
        _pending_onboardings(user_id, expert_ids),
    )
    items_by_expert: dict[str, list[ExpertSetupItem]] = defaultdict(list)
    for item in setup_items:
        items_by_expert[item.expert_id].append(item)
    return ExpertFirstJobsResponse(
        items=[
            compose_first_job(
                expert,
                day_ones.get(expert.source_template_id or "", []),
                items_by_expert[expert.id],
                sessions.get(expert.id),
                onboardings.get(expert.id),
            )
            for expert in experts
        ]
    )


def compose_first_job(
    expert: Expert,
    template_day_one: list[ExpertDayOneItem],
    setup_items: list[ExpertSetupItem],
    latest_session: LatestExpertSession | None,
    pending_onboarding: str | None,
) -> ExpertFirstJob:
    """The card for one expert. ``pending_onboarding`` is the id of the chat
    whose intake card still waits on the user, if any."""
    providers = _missing_providers(setup_items)
    status, detail = _status(expert, providers, latest_session, pending_onboarding)
    first = template_day_one[0] if template_day_one else None
    session_id = latest_session.session_id if latest_session else None
    return ExpertFirstJob(
        expert_id=expert.id,
        name=expert.name,
        role=expert.role,
        avatar_url=expert.avatar_url,
        title=first.title if first else f"Meet {expert.name} and pick a first job",
        description=first.description if first else "",
        status=status,
        detail=detail,
        providers=providers,
        # The waiting card is where the user's answer goes, even when a later
        # chat with the same expert is more recent.
        session_id=pending_onboarding if status == "needs_you" else session_id,
    )


def _status(
    expert: Expert,
    providers: list[str],
    latest_session: LatestExpertSession | None,
    pending_onboarding: str | None,
) -> tuple[ExpertFirstJobStatus, str]:
    if expert.setup_status == "installing":
        return "installing", "Setting up"
    if expert.setup_status == "failed":
        return "needs_setup", "Setup failed"
    if providers:
        return "needs_setup", f"Needs {', '.join(providers)}"
    if latest_session and latest_session.chat_status in _BUSY_CHAT_STATUSES:
        return "working", "Working on it now"
    if pending_onboarding:
        return "needs_you", "Waiting for your answer"
    if latest_session:
        return "ready", "Ready for review"
    return "open", "Not started yet"


def _missing_providers(setup_items: list[ExpertSetupItem]) -> list[str]:
    """Providers to connect or allow, each once, in first-seen order."""
    return list(
        dict.fromkeys(
            provider
            for item in setup_items
            if item.resolution in _CONNECTION_RESOLUTIONS
            for provider in item.providers
        )
    )


async def _template_day_ones(
    template_ids: list[str],
) -> dict[str, list[ExpertDayOneItem]]:
    if not template_ids:
        return {}
    rows = await prisma.models.Expert.prisma().find_many(
        where={"id": {"in": template_ids}, "isTemplate": True}
    )
    return {row.id: decode_day_one(row.dayOne) for row in rows}


async def _latest_sessions(
    user_id: str, expert_ids: list[str]
) -> dict[str, LatestExpertSession]:
    rows = await query_raw_with_schema(
        f"""
        SELECT DISTINCT ON ("expertId")
          "expertId" AS expert_id, id AS session_id, "chatStatus" AS chat_status
        FROM {{schema_prefix}}"ChatSession"
        WHERE "userId" = $1 AND "expertId" = ANY($2::text[])
          AND {exclude_dream_sessions_sql()}
        ORDER BY "expertId", "updatedAt" DESC, id DESC
        """,
        user_id,
        expert_ids,
        model=LatestExpertSession,
    )
    return {row.expert_id: row for row in rows}


async def _pending_onboardings(user_id: str, expert_ids: list[str]) -> dict[str, str]:
    """Per expert, the chat whose intake card still waits on the user. The
    same read as ``get_expert_onboarding``, for every expert at once."""
    messages = await query_raw_with_schema(
        """
        SELECT s."expertId" AS expert_id, m."sessionId" AS session_id, m.role,
          CASE
            WHEN m.role = 'tool' OR m.content = $3
              OR m.content LIKE '**Here are my answers:**%'
            THEN m.content
            ELSE ''
          END AS content
        FROM {schema_prefix}"ChatMessage" m
        JOIN {schema_prefix}"ChatSession" s ON s.id = m."sessionId"
        WHERE s."userId" = $1 AND s."expertId" = ANY($2::text[])
          AND (
            (m.role = 'tool' AND m.content LIKE '%"expert_onboarding"%')
            OR m.role = 'user'
          )
        ORDER BY m."createdAt", m.sequence
        """,
        user_id,
        expert_ids,
        onboarding.SKIP_MESSAGE,
        model=ExpertOnboardingMessage,
    )
    by_expert: dict[str, list[OnboardingMessage]] = defaultdict(list)
    for message in messages:
        by_expert[message.expert_id].append(message)
    pending = {
        expert_id: onboarding.pending_onboarding(expert_messages, expert_id)
        for expert_id, expert_messages in by_expert.items()
    }
    return {expert_id: session for expert_id, session in pending.items() if session}
