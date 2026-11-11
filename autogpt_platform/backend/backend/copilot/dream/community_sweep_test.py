"""The hourly community sweep: which scopes it rebuilds, what an unchanged
graph costs, and that it stamps every scope it takes.

The rebuild is the real ``rebuild_communities_for_user`` over a fake graph,
so its activity gate decides what is unchanged; the registry is the fake
from ``registry_fakes_test.py``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest

from backend.copilot.graphiti import communities
from backend.copilot.graphiti.scope import MemoryScope

from . import community_sweep
from .registry_fakes_test import PARIS, PAUSED, RegistryEnv, install_registry_env

NOW = datetime.now(UTC)
LONG_AGO = NOW - timedelta(days=30)


class FakeGraphs:
    """The FalkorDB side of a rebuild: episode counts per group, and the
    groups whose communities were built."""

    def __init__(self) -> None:
        self.episodes: dict[str, int] = {}
        self.built: list[str] = []
        self.failing: set[str] = set()

    def client(self, group_id: str) -> Any:
        graphs = self

        class Driver:
            async def execute_query(self, query: str, **params: Any) -> tuple:
                if "Episodic" in query:
                    total = graphs.episodes.get(group_id, 0)
                    rows = [{"latest": NOW if total else None, "total": total}]
                    return rows, None, None
                if "Community" in query and "RETURN" in query:
                    return [{"latest": None, "total": 0}], None, None
                return [], None, None

        class Client:
            graph_driver = Driver()

            async def build_communities(self, group_ids: list[str]) -> tuple:
                if group_id in graphs.failing:
                    raise RuntimeError("falkordb down")
                graphs.built.extend(group_ids)
                return [], []

        return Client()


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> RegistryEnv:
    env = install_registry_env(monkeypatch)
    monkeypatch.setattr(community_sweep, "memory_schedule_db", lambda: env.db)
    return env


@pytest.fixture
def graphs(monkeypatch: pytest.MonkeyPatch) -> FakeGraphs:
    graphs = FakeGraphs()
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)

    async def get_client(group_id: str) -> Any:
        return graphs.client(group_id)

    monkeypatch.setattr(communities, "get_redis_async", AsyncMock(return_value=redis))
    monkeypatch.setattr(communities, "get_graphiti_client", get_client)
    monkeypatch.setattr(
        communities.graphiti_config, "community_rebuild_use_flex_tier", False
    )
    return graphs


@pytest.fixture
def banned(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """The owner's refusal: None, or "banned" or "closed" (account deletion)."""
    check = AsyncMock(return_value=None)
    monkeypatch.setattr(community_sweep, "account_refusal", check)
    return check


async def active_expert(user_id: str, expert_id: str) -> str:
    return "active"


async def scope_with(env: RegistryEnv, user_id: str, last_run: datetime | None):
    scope = MemoryScope.for_user(user_id)
    await env.db.claim_scope_schedule(user_id, scope.scope_key, None, PARIS)
    env.db.update(scope.scope_key, last_community_run_at=last_run)
    return scope


def due_listing(env: RegistryEnv, monkeypatch, keys: list[str]) -> AsyncMock:
    """The registry's due listing, answering ``keys``' rows in that order."""
    listing = AsyncMock(side_effect=lambda **_: [env.db.rows[key] for key in keys])
    monkeypatch.setattr(env.db, "list_due_community_scopes", listing, raising=False)
    return listing


def last_run(env: RegistryEnv, scope: MemoryScope) -> datetime | None:
    return env.db.rows[scope.scope_key].last_community_run_at


async def sweep() -> community_sweep.SweepReport:
    return await community_sweep.rebuild_due_communities(active_expert, 60)


@pytest.mark.asyncio
async def test_a_sweep_asks_for_one_bounded_batch_of_week_old_scopes(
    env, graphs, banned, monkeypatch
):
    monkeypatch.setattr(community_sweep, "SWEEP_BATCH", 2)
    scopes = [await scope_with(env, f"user-{n}", LONG_AGO) for n in range(2)]
    listing = due_listing(env, monkeypatch, [scope.scope_key for scope in scopes])

    report = await sweep()

    (call,) = listing.await_args_list
    assert call.kwargs["limit"] == 2
    week_ago = datetime.now(UTC) - timedelta(days=7)
    assert abs(call.kwargs["due_before"] - week_ago) < timedelta(minutes=1)
    assert report.due == 2


@pytest.mark.asyncio
async def test_an_unchanged_graph_is_stamped_without_any_owner_check(
    env, graphs, banned, monkeypatch
):
    """Most scopes have no episodes: their pass asks neither the ban nor the
    registry row, and moves them a week on."""
    scopes = [await scope_with(env, f"user-{n}", LONG_AGO) for n in range(3)]
    due_listing(env, monkeypatch, [scope.scope_key for scope in scopes])
    gate = AsyncMock(wraps=community_sweep.memory_scope_may_fire)
    monkeypatch.setattr(community_sweep, "memory_scope_may_fire", gate)

    report = await sweep()

    assert report.outcomes == {"unchanged": 3}
    assert all(last_run(env, scope) > LONG_AGO for scope in scopes)
    banned.assert_not_awaited()
    gate.assert_not_awaited()
    assert graphs.built == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "refusal, paused, outcome",
    [
        (None, False, "built"),
        ("banned", False, "refused"),
        ("closed", False, "refused"),
        (None, True, "refused"),
    ],
)
async def test_a_changed_graph_is_rebuilt_only_once_its_owner_checks_pass(
    env, graphs, banned, monkeypatch, refusal, paused, outcome
):
    """A closed account (its owner asked to delete it) is refused like a
    banned one, for its 14 days and after."""
    scope = await scope_with(env, "user-1", LONG_AGO)
    graphs.episodes[scope.group_id] = 5
    due_listing(env, monkeypatch, [scope.scope_key])
    banned.return_value = refusal
    if paused:
        # Paused between the listing and the rebuild: the row is read again.
        env.db.update(scope.scope_key, state=PAUSED)

    report = await sweep()

    assert report.outcomes == {outcome: 1}
    assert graphs.built == ([scope.group_id] if outcome == "built" else [])
    assert last_run(env, scope) > LONG_AGO


@pytest.mark.asyncio
async def test_a_failed_rebuild_waits_a_week_like_the_rest(
    env, graphs, banned, monkeypatch
):
    """Retried every hour, a scope that keeps failing would stay at the head
    of every batch."""
    failing, fine = [await scope_with(env, f"user-{n}", LONG_AGO) for n in range(2)]
    for scope in (failing, fine):
        graphs.episodes[scope.group_id] = 5
    graphs.failing.add(failing.group_id)
    due_listing(env, monkeypatch, [failing.scope_key, fine.scope_key])

    report = await sweep()

    assert report.outcomes == {"failed": 1, "built": 1}
    assert last_run(env, failing) > LONG_AGO
    assert last_run(env, fine) > LONG_AGO


@pytest.mark.asyncio
async def test_a_rebuild_past_its_timeout_is_left_to_finish(env, banned, monkeypatch):
    """The sweep stops waiting and moves the scope a week on; the rebuild
    itself runs on to the end rather than stopping between its delete and
    build."""
    scope = await scope_with(env, "user-1", LONG_AGO)
    due_listing(env, monkeypatch, [scope.scope_key])
    release, finished = asyncio.Event(), asyncio.Event()

    async def slow_rebuild(*args: Any, **kwargs: Any) -> dict:
        await release.wait()
        finished.set()
        return {"error": None, "skipped": False}

    monkeypatch.setattr(community_sweep, "rebuild_communities_for_user", slow_rebuild)

    report = await community_sweep.rebuild_due_communities(active_expert, 0.05)
    release.set()
    await asyncio.wait_for(finished.wait(), 5)

    assert report.outcomes == {"failed": 1}
    assert last_run(env, scope) > LONG_AGO
