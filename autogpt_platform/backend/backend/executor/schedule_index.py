"""Queryable sidecar index for the scheduler's EXECUTION jobstore.

APScheduler's ``SQLAlchemyJobStore`` persists each job as a pickled blob, so
the only SQL-level filters it supports are ``id`` and ``next_run_time`` —
answering "which schedules does user X have?" means loading and unpickling
*every* row. On a busy deployment that full scan takes tens of seconds and
made ``get_graph_execution_schedules`` the dominant cost of the library
page (30s+ proxy timeouts surfacing as 504s — BUILDER-3A7).

This module maintains a companion table holding just the filterable columns
(user, kind, graph, session, org/team/expert) keyed by job id. It is a
*candidate* index, never an authority:

- Reads use it only to narrow the job-id set; every candidate is then
  loaded from the real jobstore and re-checked against the same in-Python
  predicate as the full-scan path. A wrong or stale index row can therefore
  never produce a wrong result — only a missing row can hide a schedule,
  and those are healed by the startup backfill / periodic reconcile in
  ``scheduler.py``.
- Rows whose job has vanished (e.g. fired one-shots that APScheduler
  auto-removes) are dropped lazily when a read trips over them.

The table is owned and auto-created by the scheduler service over the same
SQLAlchemy engine as the jobstore itself — mirroring how APScheduler manages
``apscheduler_jobs`` — rather than via Prisma, which the scheduler process
never connects to.

Writes are plain delete+insert inside one transaction (portable across
Postgres and the SQLite used in unit tests); schedule mutations are rare
enough that dialect-native upserts aren't worth the coupling.
"""

import logging
from typing import Optional, Sequence

from pydantic import BaseModel
from sqlalchemy import Column, MetaData, String, Table, delete, insert, or_, select
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Matches APScheduler's default id column length so every job id that fits
# the jobstore fits the index.
_JOB_ID_LENGTH = 191


class ScheduleIndexEntry(BaseModel):
    """The filterable identity of one schedule, keyed by its jobstore id."""

    job_id: str
    user_id: str
    kind: str
    graph_id: Optional[str] = None
    session_id: Optional[str] = None
    # Normalized: legacy rows carry ``organization_id=""`` which must never
    # match an org-scoped query, so empty strings are stored as NULL.
    organization_id: Optional[str] = None
    team_id: Optional[str] = None
    expert_id: Optional[str] = None


class ScheduleIndex:
    def __init__(self, engine: Engine, schema: str | None = None):
        self._engine = engine
        self._metadata = MetaData(schema=schema)
        self._table = Table(
            "apscheduler_jobs_index",
            self._metadata,
            Column("job_id", String(_JOB_ID_LENGTH), primary_key=True),
            Column("user_id", String, nullable=False, index=True),
            Column("kind", String, nullable=False),
            Column("graph_id", String, nullable=True, index=True),
            Column("session_id", String, nullable=True, index=True),
            Column("organization_id", String, nullable=True, index=True),
            Column("team_id", String, nullable=True),
            Column("expert_id", String, nullable=True),
        )

    def ensure_table(self) -> None:
        """Create the table if missing (same auto-DDL model as APScheduler)."""
        self._metadata.create_all(self._engine, checkfirst=True)

    def upsert(self, entry: ScheduleIndexEntry) -> None:
        self.upsert_many([entry])

    def upsert_many(self, entries: Sequence[ScheduleIndexEntry]) -> None:
        if not entries:
            return
        rows = [
            {**e.model_dump(), "organization_id": e.organization_id or None}
            for e in entries
        ]
        with self._engine.begin() as conn:
            conn.execute(
                delete(self._table).where(
                    self._table.c.job_id.in_([r["job_id"] for r in rows])
                )
            )
            conn.execute(insert(self._table), rows)

    def delete(self, job_id: str) -> None:
        self.delete_many([job_id])

    def delete_many(self, job_ids: Sequence[str]) -> None:
        if not job_ids:
            return
        with self._engine.begin() as conn:
            conn.execute(
                delete(self._table).where(self._table.c.job_id.in_(list(job_ids)))
            )

    def all_job_ids(self) -> set[str]:
        with self._engine.connect() as conn:
            return {row[0] for row in conn.execute(select(self._table.c.job_id))}

    def candidate_job_ids(
        self,
        *,
        user_id: str | None = None,
        graph_id: str | None = None,
        session_id: str | None = None,
        kind: str | None = None,
        organization_id: str | None = None,
    ) -> list[str] | None:
        """Job ids that *may* satisfy the given filters, or ``None`` when no
        identity filter was given (the trusted-global case, which must keep
        seeing rows the index doesn't track).

        The set is a deliberate superset of the exact visibility rules:
        org-scoped reads return every row tagged with the org (team and
        expert trimming happens in the caller's predicate, which re-checks
        each loaded job), because encoding those rules here would duplicate
        the authorization logic in two places.
        """
        clauses = []
        if kind is not None:
            clauses.append(self._table.c.kind == kind)
        if graph_id is not None:
            clauses.append(self._table.c.graph_id == graph_id)
        if session_id is not None:
            clauses.append(self._table.c.session_id == session_id)
        ownership = []
        if user_id is not None:
            ownership.append(self._table.c.user_id == user_id)
        if organization_id is not None:
            ownership.append(self._table.c.organization_id == organization_id)
        if ownership:
            clauses.append(or_(*ownership))

        if not (graph_id or session_id or ownership):
            # ``kind`` alone isn't an identity filter — a kind-only read is
            # a global listing and must include un-indexed rows.
            return None

        query = select(self._table.c.job_id)
        for clause in clauses:
            query = query.where(clause)
        with self._engine.connect() as conn:
            return [row[0] for row in conn.execute(query)]
