"""Shared fakes for the schedule registry tests (no tests of its own).

``install_registry_env`` swaps the registry's data module, scheduler client,
flags and timezone lookup for in-memory fakes with the real ownership rules;
``Hang`` and ``HangingTransport`` stand in for a dependency that never
answers, recording whether the caller cancelled it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from prisma.enums import MemoryScopeScheduleState

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.memory_schedule import MemoryScopeSchedule
from backend.util.exceptions import NotAuthorizedError
from backend.util.feature_flag import Flag
from backend.util.service import get_service_client

from . import deadline, registry, scope_crons, scope_jobs

ACTIVE = MemoryScopeScheduleState.ACTIVE
PAUSED = MemoryScopeScheduleState.PAUSED
WIPED = MemoryScopeScheduleState.WIPED

USER = "user-1"
ACCOUNT = MemoryScope.for_user(USER)
EXPERT = MemoryScope.for_expert(USER, "expert-1")
OTHER_EXPERT = MemoryScope.for_expert(USER, "expert-2")
PARIS = "Europe/Paris"
TOKYO = "Asia/Tokyo"

# Short enough to keep deadline tests quick; long enough for a fake to answer.
SHORT_DEADLINE = 0.05


def job_ids(scope: MemoryScope) -> set[str]:
    return {
        f"community_rebuild_{scope.scope_key}",
        f"dream_nightly_batch_{scope.scope_key}",
    }


class FakeRegistryDB:
    """``backend.data.memory_schedule`` in memory, with its ownership rules."""

    def __init__(self) -> None:
        self.rows: dict[str, MemoryScopeSchedule] = {}
        self.fail = False

    async def get_scope_schedule(
        self, user_id: str, scope_key: str
    ) -> MemoryScopeSchedule | None:
        if self.fail:
            raise ConnectionError("db down")
        row = self.rows.get(scope_key)
        return row if row is not None and row.user_id == user_id else None

    async def claim_scope_schedule(
        self,
        user_id: str,
        scope_key: str,
        expert_id: str | None,
        user_timezone: str,
        state: MemoryScopeScheduleState = ACTIVE,
    ) -> MemoryScopeSchedule:
        if self.fail:
            raise ConnectionError("db down")
        if scope_key not in self.rows:
            now = datetime.now(timezone.utc)
            self.rows[scope_key] = MemoryScopeSchedule(
                scope_key=scope_key,
                user_id=user_id,
                expert_id=expert_id,
                timezone=user_timezone,
                state=state,
                community_job_id=None,
                nightly_job_id=None,
                last_nightly_run_at=None,
                last_community_run_at=None,
                created_at=now,
                updated_at=now,
            )
        if self.rows[scope_key].user_id != user_id:
            raise NotAuthorizedError("another user's scope")
        return self.rows[scope_key]

    async def record_scope_jobs(
        self,
        user_id: str,
        scope_key: str,
        *,
        user_timezone: str,
        community_job_id: str | None,
        nightly_job_id: str | None,
    ) -> bool:
        row = await self.get_scope_schedule(user_id, scope_key)
        if row is None or row.state != ACTIVE:
            return False
        self.update(
            scope_key,
            timezone=user_timezone,
            community_job_id=community_job_id,
            nightly_job_id=nightly_job_id,
        )
        return True

    async def set_scope_state(
        self,
        user_id: str,
        scope_key: str,
        state: MemoryScopeScheduleState,
        *,
        only_from: Sequence[MemoryScopeScheduleState] | None = None,
    ) -> bool:
        row = await self.get_scope_schedule(user_id, scope_key)
        if row is None or (only_from is not None and row.state not in only_from):
            return False
        fields: dict[str, Any] = {"state": state}
        if state != ACTIVE:
            fields |= {"community_job_id": None, "nightly_job_id": None}
        self.update(scope_key, **fields)
        return True

    async def forget_scope_job(self, user_id: str, scope_key: str, job_id: str) -> bool:
        row = await self.get_scope_schedule(user_id, scope_key)
        if row is None:
            return False
        fields: dict[str, Any] = {}
        if row.community_job_id == job_id:
            fields["community_job_id"] = None
        if row.nightly_job_id == job_id:
            fields["nightly_job_id"] = None
        self.update(scope_key, **fields)
        return bool(fields)

    async def record_scope_run(self, user_id: str, scope_key: str, kind: str) -> bool:
        if await self.get_scope_schedule(user_id, scope_key) is None:
            return False
        field = "last_nightly_run_at" if kind == "nightly" else "last_community_run_at"
        self.update(scope_key, **{field: datetime.now(timezone.utc)})
        return True

    async def list_user_scope_schedules(
        self, user_id: str
    ) -> list[MemoryScopeSchedule]:
        return [row for row in self.rows.values() if row.user_id == user_id]

    def update(self, scope_key: str, **fields: Any) -> None:
        self.rows[scope_key] = self.rows[scope_key].model_copy(update=fields)


class FakeScheduler:
    """The three memory-cron RPCs of ``SchedulerClient``, in memory."""

    def __init__(self) -> None:
        self.jobs: dict[str, str] = {}  # job id -> timezone it runs in
        self.add_scope_community_rebuild_schedule = AsyncMock(
            side_effect=self._adder("community_rebuild")
        )
        self.add_scope_nightly_batch_schedule = AsyncMock(
            side_effect=self._adder("dream_nightly_batch")
        )
        self.remove_scope_memory_jobs = AsyncMock(side_effect=self._remove)

    def _adder(self, prefix: str):
        async def add(*, scope: MemoryScope, user_timezone: str) -> dict:
            job_id = f"{prefix}_{scope.scope_key}"
            self.jobs[job_id] = user_timezone
            return {"id": job_id, "scope_key": scope.scope_key, "next_run_time": None}

        return add

    async def _remove(self, *, scope: MemoryScope) -> list[str]:
        removed = [job_id for job_id in job_ids(scope) if job_id in self.jobs]
        for job_id in removed:
            del self.jobs[job_id]
        return removed


class RegistryEnv:
    def __init__(self) -> None:
        self.db = FakeRegistryDB()
        self.scheduler = FakeScheduler()
        self.flags = {
            Flag.GRAPHITI_COMMUNITIES_ENABLED: True,
            Flag.DREAM_PASS_ENABLED: True,
        }
        self.timezone: str | None = PARIS

    def adds(self) -> int:
        return (
            self.scheduler.add_scope_community_rebuild_schedule.await_count
            + self.scheduler.add_scope_nightly_batch_schedule.await_count
        )


def install_registry_env(monkeypatch: pytest.MonkeyPatch) -> RegistryEnv:
    """Point the registry modules at a fresh set of fakes."""
    env = RegistryEnv()

    async def flag(flag: Flag, user_id: str) -> bool:
        return env.flags.get(flag, False)

    async def owner_timezone(user_id: str) -> str | None:
        return env.timezone

    for module in (registry, scope_jobs, scope_crons):
        monkeypatch.setattr(module, "memory_schedule_db", lambda: env.db)
    monkeypatch.setattr(scope_jobs, "get_scheduler_client", lambda: env.scheduler)
    monkeypatch.setattr(scope_jobs, "is_feature_enabled", flag)
    monkeypatch.setattr(registry, "resolve_user_timezone", owner_timezone)
    return env


def use_short_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deadline, "REGISTRY_CALL_TIMEOUT_SECONDS", SHORT_DEADLINE)
    monkeypatch.setattr(deadline, "BRIDGE_MARGIN_SECONDS", SHORT_DEADLINE)


class Hang:
    """A dependency call that never returns, counting the calls that started
    and the ones their caller cancelled."""

    def __init__(self) -> None:
        self.started = 0
        self.cancelled = 0

    async def forever(self, *args: Any, **kwargs: Any) -> Any:
        self.started += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


class HangingTransport:
    """An HTTP transport for a real RPC client whose requests never answer,
    recording each request path that reached it and each one cancelled."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.cancelled: list[str] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.append(request.url.path)
            raise
        raise AssertionError("unreachable")

    def client(self, client_type: Any, *, request_retry: bool) -> Any:
        """A real service client of ``client_type`` whose calls on the
        running loop go through this transport."""
        rpc = get_service_client(client_type, request_retry=request_retry)
        rpc._async_clients[asyncio.get_running_loop()] = httpx.AsyncClient(
            transport=httpx.MockTransport(self.handle), base_url="http://hanging"
        )
        return rpc
