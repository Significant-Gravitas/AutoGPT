from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.dream.nightly_batch import NightlyBatchResult

from . import maintenance
from .scope import MemoryScope

_PERSONAL = MemoryScope.for_user("owner")
_PERSONAL_EXPERT = MemoryScope.for_expert("owner", "writer")
_ORG = MemoryScope.build("owner", organization_id="denton")
_ORG_EXPERT = MemoryScope.build("owner", "reviewer", "denton")


def _result(**updates) -> NightlyBatchResult:
    return NightlyBatchResult(
        user_id="owner",
        nightly_id="test",
        started_at=datetime.now(timezone.utc),
        **updates,
    )


@pytest.fixture
def persisted_scopes(monkeypatch):
    store = MagicMock()
    store.list_memory_scopes = AsyncMock(
        return_value=[_PERSONAL, _PERSONAL_EXPERT, _ORG, _ORG_EXPERT]
    )
    monkeypatch.setattr(maintenance, "memory_db", lambda: store)


@pytest.fixture(autouse=True)
def experts(monkeypatch) -> MagicMock:
    """Every expert is active unless a test says otherwise."""
    store = MagicMock()
    store.get_expert = AsyncMock(
        return_value=SimpleNamespace(is_archived=False, schedules_paused_at=None)
    )
    monkeypatch.setattr(maintenance, "experts_db", lambda: store)
    return store


@pytest.fixture
def authorized(monkeypatch) -> list[MemoryScope]:
    seen: list[MemoryScope] = []

    class Lease:
        async def run(self, action):
            return await action

    @asynccontextmanager
    async def access(scope):
        seen.append(scope)
        yield Lease()

    monkeypatch.setattr(maintenance, "memory_scope_access", access)
    return seen


@pytest.mark.asyncio
@pytest.mark.usefixtures("persisted_scopes")
async def test_the_account_cron_also_runs_each_saved_organizations_scope(
    monkeypatch, authorized
):
    """Personal scopes have crons of their own; the account's cron runs its
    own scope and then each saved organization's, each authorized."""
    submit = AsyncMock(side_effect=lambda user_id, **kwargs: _result())
    monkeypatch.setattr(maintenance, "run_nightly_batch_submit", submit)

    result = await maintenance.run_all_memory_nightly("owner", trigger="admin")

    assert [call.kwargs for call in submit.await_args_list] == [
        {"expert_id": None, "trigger": "admin"},
        *(
            {
                "expert_id": scope.expert_id,
                "organization_id": scope.organization_id,
                "include_learning": False,
                "trigger": "admin",
            }
            for scope in (_ORG, _ORG_EXPERT)
        ),
    ]
    assert set(result.scope_results) == {_ORG.scope_key, _ORG_EXPERT.scope_key}
    assert authorized == [_ORG, _ORG_EXPERT]
    assert result.error is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("persisted_scopes")
async def test_an_experts_cron_runs_only_its_own_scope(monkeypatch, authorized):
    submit = AsyncMock(side_effect=lambda user_id, **kwargs: _result())
    monkeypatch.setattr(maintenance, "run_nightly_batch_submit", submit)

    await maintenance.run_all_memory_nightly("owner", expert_id="writer")

    submit.assert_awaited_once_with("owner", expert_id="writer", trigger="cron")
    assert authorized == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("persisted_scopes")
async def test_a_budget_stop_stops_every_organizations_scope(monkeypatch, authorized):
    submit = AsyncMock(
        return_value=_result(skipped=True, skip_reason="insufficient_credits")
    )
    monkeypatch.setattr(maintenance, "run_nightly_batch_submit", submit)

    result = await maintenance.run_all_memory_nightly("owner")

    assert result.skipped
    submit.assert_awaited_once()
    assert authorized == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("persisted_scopes")
async def test_a_failed_organization_scope_marks_the_run_errored(
    monkeypatch, authorized
):
    async def submit(user_id, **kwargs):
        if kwargs.get("organization_id"):
            return _result(error="dream: crashed")
        return _result()

    monkeypatch.setattr(maintenance, "run_nightly_batch_submit", submit)

    result = await maintenance.run_all_memory_nightly("owner")

    assert result.error == maintenance.SCOPES_FAILED


@pytest.mark.asyncio
@pytest.mark.usefixtures("persisted_scopes")
async def test_a_revoked_organization_is_not_maintained(monkeypatch):
    @asynccontextmanager
    async def revoked(scope):
        yield None

    submit = AsyncMock(side_effect=lambda user_id, **kwargs: _result())
    rebuild = AsyncMock(return_value={"error": None})
    monkeypatch.setattr(maintenance, "memory_scope_access", revoked)
    monkeypatch.setattr(maintenance, "run_nightly_batch_submit", submit)
    monkeypatch.setattr(maintenance, "rebuild_communities_for_user", rebuild)

    result = await maintenance.run_all_memory_nightly("owner")
    communities = await maintenance.rebuild_all_memory_communities("owner")

    submit.assert_awaited_once_with("owner", expert_id=None, trigger="cron")
    rebuild.assert_awaited_once_with("owner", expert_id=None, force=False)
    assert result.scope_results == {}
    assert communities["scopes"] == {}


@pytest.mark.asyncio
@pytest.mark.usefixtures("persisted_scopes")
async def test_community_rebuilds_follow_the_same_scopes(monkeypatch, authorized):
    rebuild = AsyncMock(return_value={"error": None})
    monkeypatch.setattr(maintenance, "rebuild_communities_for_user", rebuild)

    result = await maintenance.rebuild_all_memory_communities("owner")

    assert [call.kwargs for call in rebuild.await_args_list] == [
        {"expert_id": None, "force": False},
        *(
            {
                "expert_id": scope.expert_id,
                "organization_id": scope.organization_id,
                "force": False,
            }
            for scope in (_ORG, _ORG_EXPERT)
        ),
    ]
    assert set(result["scopes"]) == {_ORG.scope_key, _ORG_EXPERT.scope_key}
    assert authorized == [_ORG, _ORG_EXPERT]


@pytest.mark.asyncio
@pytest.mark.usefixtures("persisted_scopes")
@pytest.mark.parametrize(
    "expert",
    [
        None,
        SimpleNamespace(is_archived=True, schedules_paused_at=None),
        SimpleNamespace(
            is_archived=False, schedules_paused_at=datetime.now(timezone.utc)
        ),
    ],
    ids=["missing", "archived", "paused"],
)
async def test_an_inactive_organization_expert_is_not_maintained(
    monkeypatch, authorized, experts, expert
):
    """An organization's expert scope runs only while the expert is active,
    as a personal expert's own crons are held: archived, paused (a budget
    breach included) or gone, it is left alone."""
    experts.get_expert = AsyncMock(return_value=expert)
    submit = AsyncMock(side_effect=lambda user_id, **kwargs: _result())
    monkeypatch.setattr(maintenance, "run_nightly_batch_submit", submit)

    result = await maintenance.run_all_memory_nightly("owner")

    assert set(result.scope_results) == {_ORG.scope_key}
    assert authorized == [_ORG]
    experts.get_expert.assert_awaited_once_with(
        "owner", "reviewer", include_workflows=False
    )
