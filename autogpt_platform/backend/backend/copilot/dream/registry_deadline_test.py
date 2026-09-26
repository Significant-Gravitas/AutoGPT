"""Every registry entry point returns within the registry deadline when a
dependency hangs, and cancels the call it gave up on.

The deadline is shortened (``use_short_deadline``). Where the dependency is an
RPC, the real DatabaseManager or scheduler client is used with a transport
that never answers, so the client's own retry and timeout layers are part of
what is bounded. The cron gate and run stamp are covered in
``scope_crons_test.py``, the API hooks in the experts hook tests.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

import httpx
import pytest

from backend.data.db_manager import DatabaseManagerAsyncClient
from backend.executor.scheduler import SchedulerClient
from backend.util.service import get_service_client

from . import deadline, registry, scope_crons, scope_jobs
from .registry_fakes_test import (
    ACCOUNT,
    EXPERT,
    USER,
    Hang,
    HangingTransport,
    RegistryEnv,
    install_registry_env,
    use_short_deadline,
)

# Wall-clock allowance for a call bounded by the short deadline.
PROMPT = 1.0


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> RegistryEnv:
    use_short_deadline(monkeypatch)
    return install_registry_env(monkeypatch)


def _use_registry_rpc(monkeypatch: pytest.MonkeyPatch, rpc: object) -> None:
    for module in (registry, scope_jobs, scope_crons):
        monkeypatch.setattr(module, "memory_schedule_db", lambda: rpc)


@pytest.mark.asyncio
async def test_ensure_is_bounded_when_the_database_manager_hangs(env, monkeypatch):
    transport = HangingTransport()
    rpc = transport.client(DatabaseManagerAsyncClient, request_retry=True)
    _use_registry_rpc(monkeypatch, rpc)
    started = time.monotonic()
    try:
        result = await registry.ensure_scope_scheduled(ACCOUNT)
    finally:
        await rpc.aclose()

    assert time.monotonic() - started < PROMPT
    assert {r["reason"] for r in result.values()} == {"registry_unavailable"}
    assert transport.cancelled == transport.paths == ["/get_scope_schedule"]


@pytest.mark.asyncio
async def test_a_retrying_database_manager_client_is_cut_off(env, monkeypatch):
    """A refused connection sends the real client into its retry backoff;
    the deadline cuts the whole retry loop off, not just one attempt."""
    attempts: list[str] = []

    def refuse(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        raise httpx.ConnectError("refused", request=request)

    rpc = get_service_client(DatabaseManagerAsyncClient, request_retry=True)
    rpc._async_clients[asyncio.get_running_loop()] = httpx.AsyncClient(
        transport=httpx.MockTransport(refuse), base_url="http://refusing"
    )
    _use_registry_rpc(monkeypatch, rpc)
    started = time.monotonic()
    try:
        result = await registry.ensure_scope_scheduled(ACCOUNT)
    finally:
        await rpc.aclose()

    assert time.monotonic() - started < PROMPT
    assert {r["reason"] for r in result.values()} == {"registry_unavailable"}
    assert attempts == ["/get_scope_schedule"]


@pytest.mark.asyncio
async def test_a_hanging_scheduler_is_cut_off_and_reported(env, monkeypatch):
    transport = HangingTransport()
    scheduler = transport.client(SchedulerClient, request_retry=False)
    monkeypatch.setattr(scope_jobs, "get_scheduler_client", lambda: scheduler)
    started = time.monotonic()
    try:
        result = await registry.ensure_scope_scheduled(ACCOUNT)
    finally:
        await scheduler.aclose()

    assert time.monotonic() - started < PROMPT
    assert {r["reason"] for r in result.values()} == {"registration_failed"}
    assert transport.cancelled == transport.paths
    assert transport.paths == [
        "/add_scope_community_rebuild_schedule",
        "/add_scope_nightly_batch_schedule",
    ]
    assert env.db.rows[USER].nightly_job_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("active", [True, False])
async def test_a_lifecycle_change_is_bounded_as_a_whole(env, monkeypatch, active):
    """Pause, resume, archive and revive run this inside an API request: the
    change as a whole returns within the deadline and cancels what hangs."""
    await registry.ensure_scope_scheduled(EXPERT)
    await registry.pause_scope(EXPERT)
    hang = Hang()
    for rpc in (
        "remove_scope_memory_jobs",
        "add_scope_community_rebuild_schedule",
        "add_scope_nightly_batch_schedule",
    ):
        monkeypatch.setattr(env.scheduler, rpc, hang.forever)
    started = time.monotonic()

    await registry.sync_expert_scope(USER, "expert-1", active=active)

    assert time.monotonic() - started < PROMPT
    # Whichever deadline fires first, nothing that started is left running.
    assert hang.started >= 1
    assert hang.cancelled == hang.started


@pytest.mark.asyncio
async def test_a_slow_resume_is_cut_off_as_a_whole_not_per_call(env, monkeypatch):
    """Every call answers well inside the deadline, so no single call times
    out, but a resume makes about eight of them: the request waits for one
    deadline, not for their sum. (A longer deadline than the other tests, so
    Windows' ~16 ms timer granularity cannot push a step past it.)"""
    monkeypatch.setattr(deadline, "REGISTRY_CALL_TIMEOUT_SECONDS", 0.2)
    await registry.pause_scope(EXPERT)
    step = 0.05
    for owner, name in (
        (env.db, "set_scope_state"),
        (env.db, "get_scope_schedule"),
        (env.db, "claim_scope_schedule"),
        (env.db, "record_scope_jobs"),
        (env.scheduler, "add_scope_community_rebuild_schedule"),
        (env.scheduler, "add_scope_nightly_batch_schedule"),
        (scope_jobs, "is_feature_enabled"),
        (registry, "resolve_user_timezone"),
    ):
        monkeypatch.setattr(owner, name, _slowed(getattr(owner, name), step))
    started = time.monotonic()

    await registry.sync_expert_scope(USER, "expert-1", active=True)

    # One deadline is 0.2 s; the eight steps alone would take 0.4 s or more.
    assert time.monotonic() - started < 0.3


def _slowed(call: Callable[..., Awaitable[Any]], seconds: float):
    async def slow(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(seconds)
        return await call(*args, **kwargs)

    return slow


@pytest.mark.asyncio
async def test_a_new_expert_registration_is_bounded_when_flags_hang(env, monkeypatch):
    hang = Hang()
    monkeypatch.setattr(scope_jobs, "is_feature_enabled", hang.forever)
    started = time.monotonic()

    await registry.ensure_expert_scheduled(USER, "expert-1")

    assert time.monotonic() - started < PROMPT
    assert hang.started == hang.cancelled == 2
    assert env.db.rows == {}


@pytest.mark.asyncio
async def test_a_timezone_change_is_bounded_when_the_listing_hangs(env, monkeypatch):
    hang = Hang()
    monkeypatch.setattr(env.db, "list_user_scope_schedules", hang.forever)
    started = time.monotonic()

    results = await registry.reregister_user(USER)

    assert time.monotonic() - started < PROMPT
    assert hang.started == hang.cancelled == 1
    assert list(results) == [USER]  # the account still goes through
