"""Gather bounded inputs for a dream pass.

All input lists are clamped per ``dream/p0-spec.md`` §2 — phase 1's
prompt is the hot path for cost, so we never let it grow unbounded
even on a power-user with thousands of episodes.

Caps (copied here so they live next to the code that enforces them):
  * Recent sessions: 10
  * Recent episodes: 50
  * Active facts (per scope): 500
  * Touch window: 14 days
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field

from backend.copilot.graphiti.client import derive_group_id
from backend.copilot.graphiti.config import graphiti_config
from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver
from backend.data.db_accessors import chat_db

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 14
MAX_EPISODES = 50
MAX_ACTIVE_FACTS = 500
MAX_RECENT_SESSIONS = 10
MAX_SESSION_BODY_BYTES = 8 * 1024  # 8 KB per p0-spec.md §2


class EpisodeRow(BaseModel):
    uuid: str
    name: str | None
    content: str | None
    source_description: str | None
    valid_at: str | None  # ISO timestamp
    created_at: str | None


class FactRow(BaseModel):
    uuid: str
    source: str | None
    target: str | None
    name: str | None
    fact: str | None
    scope: str | None
    confidence: float | None
    status: str | None
    created_at: str | None


class SessionRow(BaseModel):
    session_id: str
    title: str | None
    created_at: datetime | None
    body: str  # truncated to MAX_SESSION_BODY_BYTES


class DreamInput(BaseModel):
    """The whole gather-step bundle handed to the phase 1 prompt."""

    user_id: str
    group_id: str
    window_start: datetime
    window_end: datetime
    episodes: list[EpisodeRow] = Field(default_factory=list)
    facts: list[FactRow] = Field(default_factory=list)
    recent_sessions: list[SessionRow] = Field(default_factory=list)
    # Convenience — phase 3 sanitizer needs to know which uuids the dream
    # is "aware of" so it can reject demotions for unknown edges.
    known_fact_uuids: set[str] = Field(default_factory=set)
    known_episode_uuids: set[str] = Field(default_factory=set)


def _open_driver(group_id: str) -> AutoGPTFalkorDriver:
    return AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
        # Indices are built by the long-lived chat-write client when the
        # user first writes a memory; the dream pass reads from an
        # already-indexed graph and shouldn't refire the
        # background-task race that produces "Buffer is closed" spam.
        build_indices=False,
    )


async def _fetch_recent_episodes(
    driver: AutoGPTFalkorDriver,
    group_id: str,
    window_start: datetime,
    limit: int,
) -> list[EpisodeRow]:
    # FalkorDB does not implement Cypher's ``datetime()`` function, so
    # we cannot use ``WHERE n.valid_at >= datetime($since)`` here — it
    # raises ``Unknown function 'datetime'``. The dream window is
    # already bounded by ``limit`` (default 50 most-recent episodes,
    # which exceeds typical 14-day activity for any one user). The
    # ``window_start`` argument is kept for caller bookkeeping but the
    # Cypher itself just relies on the ORDER BY + LIMIT clamp.
    _ = window_start
    try:
        result = await driver.execute_query(
            """
            MATCH (n:Episodic {group_id: $g})
            RETURN n.uuid AS uuid,
                   n.name AS name,
                   n.content AS content,
                   n.source_description AS source_description,
                   toString(n.valid_at) AS valid_at,
                   toString(n.created_at) AS created_at
            ORDER BY n.valid_at DESC
            LIMIT $limit
            """,
            g=group_id,
            limit=limit,
        )
    except Exception:
        logger.warning(
            "Failed to fetch recent episodes for group %s — treating as empty",
            group_id[:12],
            exc_info=True,
        )
        return []
    rows = result[0] if result else []
    return [
        EpisodeRow(
            uuid=str(r.get("uuid", "")),
            name=r.get("name"),
            content=r.get("content"),
            source_description=r.get("source_description"),
            valid_at=r.get("valid_at"),
            created_at=r.get("created_at"),
        )
        for r in rows
    ]


async def _fetch_active_facts(
    driver: AutoGPTFalkorDriver,
    group_id: str,
    limit: int,
) -> list[FactRow]:
    try:
        result = await driver.execute_query(
            """
            MATCH (src:Entity)-[e:RELATES_TO {group_id: $g}]->(tgt:Entity)
            WHERE (e.status IS NULL OR e.status = 'active')
              AND (e.expired_at IS NULL)
            RETURN e.uuid AS uuid,
                   src.name AS source,
                   tgt.name AS target,
                   e.name AS name,
                   e.fact AS fact,
                   e.scope AS scope,
                   e.confidence AS confidence,
                   e.status AS status,
                   toString(e.created_at) AS created_at
            ORDER BY e.created_at DESC
            LIMIT $limit
            """,
            g=group_id,
            limit=limit,
        )
    except Exception:
        logger.warning(
            "Failed to fetch active facts for group %s — treating as empty",
            group_id[:12],
            exc_info=True,
        )
        return []
    rows = result[0] if result else []
    return [
        FactRow(
            uuid=str(r.get("uuid", "")),
            source=r.get("source"),
            target=r.get("target"),
            name=r.get("name"),
            fact=r.get("fact"),
            scope=r.get("scope"),
            confidence=r.get("confidence"),
            status=r.get("status"),
            created_at=r.get("created_at"),
        )
        for r in rows
    ]


async def _fetch_recent_sessions(
    user_id: str,
    window_start: datetime,
    limit: int,
) -> list[SessionRow]:
    """Pull the most recent N chat sessions and their first chunk of content.

    Routes through ``chat_db()`` so the dream pass — which runs in the
    Scheduler subprocess — uses the DatabaseManager RPC when Prisma
    isn't directly connected, and the in-process Postgres path when
    it is. Direct ``PrismaChatSession.prisma()`` calls fail with
    ``ClientNotConnectedError`` in the Scheduler subprocess otherwise.

    We only need a flavour-snapshot of each session, not the full
    message list — phase 1's prompt cares about *what was discussed
    recently* as context for consolidation, not about replaying it.
    The ``window_start`` argument is informational here; the underlying
    ``get_user_chat_sessions`` orders by most-recent and the ``limit``
    clamp does the bounding.
    """
    _ = window_start
    try:
        sessions = await chat_db().get_user_chat_sessions(user_id, limit=limit)
    except Exception:
        logger.warning(
            "Failed to fetch recent sessions for user %s",
            user_id[:12],
            exc_info=True,
        )
        return []

    out: list[SessionRow] = []
    for s in sessions:
        # ``ChatSessionInfo`` exposes ``session_id`` (not ``id``) and
        # carries no messages (it's a summary). Fetch the body lazily
        # per-session via the paginated reader so the prompt has
        # actual content. One round-trip per session is acceptable at
        # limit=10; a bulk variant can replace this when chat_db
        # gains one.
        sid = s.session_id
        messages: list = []
        try:
            paginated = await chat_db().get_chat_messages_paginated(
                session_id=sid, limit=20, user_id=user_id
            )
            if paginated is not None:
                messages = list(getattr(paginated, "messages", []) or [])
        except Exception:
            logger.debug(
                "Failed to fetch messages for session %s — body left empty",
                sid[:12],
                exc_info=True,
            )
        body_parts: list[str] = []
        running_len = 0
        for m in messages:
            line = f"{m.role}: {m.content or ''}"
            running_len += len(line) + 1
            body_parts.append(line)
            if running_len >= MAX_SESSION_BODY_BYTES:
                break
        body = "\n".join(body_parts)[:MAX_SESSION_BODY_BYTES]
        out.append(
            SessionRow(
                session_id=sid,
                title=s.title,
                created_at=getattr(s, "createdAt", None)
                or getattr(s, "created_at", None),
                body=body,
            )
        )
    return out


async def gather_dream_input(
    user_id: str,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    max_episodes: int = MAX_EPISODES,
    max_facts: int = MAX_ACTIVE_FACTS,
    max_sessions: int = MAX_RECENT_SESSIONS,
) -> DreamInput:
    """Build the full bounded input bundle for one dream pass.

    All clamps are enforced here, not in the caller. If any one of
    the three sources fails (Cypher error, Prisma timeout) the others
    still proceed — a partial dream is better than no dream.
    """
    group_id = derive_group_id(user_id)
    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(days=window_days)

    driver = _open_driver(group_id)
    try:
        episodes = await _fetch_recent_episodes(
            driver, group_id, window_start, max_episodes
        )
        facts = await _fetch_active_facts(driver, group_id, max_facts)
    finally:
        await driver.close()

    sessions = await _fetch_recent_sessions(user_id, window_start, max_sessions)

    return DreamInput(
        user_id=user_id,
        group_id=group_id,
        window_start=window_start,
        window_end=window_end,
        episodes=episodes,
        facts=facts,
        recent_sessions=sessions,
        known_fact_uuids={f.uuid for f in facts},
        known_episode_uuids={e.uuid for e in episodes},
    )
