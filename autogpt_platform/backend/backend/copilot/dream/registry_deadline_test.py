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
from unittest.mock import AsyncMock

import httpx
import pytest

from backend.data.db_manager import DatabaseManagerAsyncClient
from backend.executor.scheduler import SchedulerClient
from backend.util.service import get_service_client

from . import deadline, registry, scheduling, scope_crons, scope_jobs
from .registry_fakes_test import (
    ACCOUNT,
    EXPERT,
    PARIS,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", ["write", "clear"])
async def test_a_marker_finishing_at_a_deadline_does_not_swallow_it(
    monkeypatch, marker
):
    """On Python 3.11 a nested ``wait_for`` that completes just as the
    enclosing deadline expires swallows the cancellation, and its caller
    runs on as if there were no deadline."""

    async def stall(*args: Any) -> None:
        await asyncio.sleep(0.02)
        time.sleep(0.05)  # the enclosing deadline passes while it finishes

    monkeypatch.setattr(scheduling, "_set_marker", stall)
    monkeypatch.setattr(scheduling, "_delete_marker", stall)
    started = time.monotonic()

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.04):
            if marker == "write":
                await scheduling.write_registration_marker(ACCOUNT, "probe", PARIS)
            else:
                await scheduling.clear_registration_marker(ACCOUNT, "probe")
            await asyncio.sleep(1)  # the next await must see the deadline

    assert time.monotonic() - started < 0.5


@pytest.mark.asyncio
async def test_a_marker_at_the_deadline_does_not_extend_a_lifecycle_change(
    env, monkeypatch
):
    """Codex's round-2 reproduction, scaled down: the community cron is
    registered just inside the deadline, its Redis marker completes right
    at it after a short loop stall, and the nightly registration would then
    hang. The change must end at the deadline; it used to register the
    nightly on a fresh deadline and record it, taking twice as long."""
    budget = 0.3
    monkeypatch.setattr(deadline, "REGISTRY_CALL_TIMEOUT_SECONDS", budget)
    loop = asyncio.get_running_loop()
    started = loop.time()
    add_community = env.scheduler.add_scope_community_rebuild_schedule

    async def community(**kwargs: Any) -> Any:
        await asyncio.sleep(started + budget - 0.1 - loop.time())
        return await add_community(**kwargs)

    async def marker(*args: Any) -> None:
        await asyncio.sleep(started + budget - 0.02 - loop.time())
        time.sleep(0.06)  # a short loop stall carries it past the deadline

    hang = Hang()
    record = AsyncMock(side_effect=env.db.record_scope_jobs)
    monkeypatch.setattr(
        env.scheduler, "add_scope_community_rebuild_schedule", community
    )
    monkeypatch.setattr(env.scheduler, "add_scope_nightly_batch_schedule", hang.forever)
    monkeypatch.setattr(scheduling, "_set_marker", marker)
    monkeypatch.setattr(env.db, "record_scope_jobs", record)

    await registry.sync_expert_scope(USER, "expert-1", active=True)

    # Nothing ran on after the deadline: no nightly on a fresh deadline (that
    # took twice the budget) and no recording of what it registered.
    assert loop.time() - started < budget + 0.25
    assert hang.cancelled == hang.started
    record.assert_not_awaited()
