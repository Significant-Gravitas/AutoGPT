"""Tests for the registry backfill: which scopes it schedules, which it
pauses, and the counts it reports. The registry functions and the listings
are mocked; ``registry_test.py`` covers what they do."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from prisma.enums import MemoryScopeScheduleState

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.memory_schedule import LiveExpertScope, MemoryScopeSchedule

from . import registry_backfill
from .registry_backfill import BackfillReport, backfill_schedules

REGISTERED = {"community_rebuild": {"id": "c"}, "dream_nightly_batch": {"id": "n"}}
ALREADY = {"community_rebuild": None, "dream_nightly_batch": None}
FLAGS_OFF = {
    "community_rebuild": {"skipped": True, "reason": "graphiti_communities_disabled"},
    "dream_nightly_batch": {"skipped": True, "reason": "dream_pass_disabled"},
}
FAILED = {
    "community_rebuild": {"skipped": True, "reason": "registration_failed"},
    "dream_nightly_batch": None,
}


def _row(user_id: str, expert_id: str | None = None) -> MemoryScopeSchedule:
    scope = MemoryScope.build(user_id, expert_id)
    now = datetime.now(timezone.utc)
    return MemoryScopeSchedule(
        scope_key=scope.scope_key,
        user_id=user_id,
        expert_id=expert_id,
        timezone="UTC",
        state=MemoryScopeScheduleState.ACTIVE,
        community_job_id=None,
        nightly_job_id=None,
        last_nightly_run_at=None,
        last_community_run_at=None,
        created_at=now,
        updated_at=now,
    )


class Backfill:
    """The backfill's inputs and the registry calls it made."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.graph_owners = ["u-graph", "u-gone"]
        self.users = {"u-graph", "u-row"}
        self.rows = [_row("u-row"), _row("u-row", "e-archived")]
        self.experts = [
            LiveExpertScope(expert_id="e-live", user_id="u-graph", paused=False),
            LiveExpertScope(expert_id="e-paused", user_id="u-graph", paused=True),
        ]
        self.ensure = AsyncMock(return_value=REGISTERED)
        self.resume = AsyncMock(return_value=ALREADY)
        self.pause = AsyncMock(return_value=True)

        async def owners() -> list[str]:
            return self.graph_owners

        async def existing(user_ids: list[str]) -> set[str]:
            return self.users & set(user_ids)

        async def rows(*, after: str | None, limit: int) -> list[MemoryScopeSchedule]:
            return [] if after else self.rows

        async def experts(*, after: str | None, limit: int) -> list[LiveExpertScope]:
            return [] if after else self.experts

        module = registry_backfill.memory_schedule
        monkeypatch.setattr(registry_backfill, "list_account_graph_owners", owners)
        monkeypatch.setattr(module, "existing_user_ids", existing)
        monkeypatch.setattr(module, "list_active_scope_schedules", rows)
        monkeypatch.setattr(module, "list_live_expert_scopes", experts)
        monkeypatch.setattr(registry_backfill, "ensure_scope_scheduled", self.ensure)
        monkeypatch.setattr(registry_backfill, "resume_scope", self.resume)
        monkeypatch.setattr(registry_backfill, "pause_scope", self.pause)

    def paused_scopes(self) -> list[MemoryScope]:
        return [call.args[0] for call in self.pause.await_args_list]


@pytest.fixture
def backfill(monkeypatch: pytest.MonkeyPatch) -> Backfill:
    return Backfill(monkeypatch)


@pytest.mark.asyncio
async def test_backfill_schedules_accounts_and_reconciles_experts(backfill):
    report = await backfill_schedules()

    assert report == BackfillReport(
        dry_run=False,
        accounts=2,
        experts=2,
        experts_to_pause=2,
        graphs_without_user=1,
        registered=2,
        already_scheduled=1,
        paused=2,
    )
    # Accounts: every graph owner with a user row, plus every ACTIVE row.
    accounts = [call.args[0] for call in backfill.ensure.await_args_list]
    assert accounts == [MemoryScope.for_user("u-graph"), MemoryScope.for_user("u-row")]
    backfill.resume.assert_awaited_once_with(
        MemoryScope.for_expert("u-graph", "e-live"), force_refresh=False
    )
    # The paused live expert and the ACTIVE row of an archived one.
    assert backfill.paused_scopes() == [
        MemoryScope.for_expert("u-graph", "e-paused"),
        MemoryScope.for_expert("u-row", "e-archived"),
    ]


@pytest.mark.asyncio
async def test_dry_run_counts_and_changes_nothing(backfill):
    report = await backfill_schedules(dry_run=True)

    assert (report.accounts, report.experts, report.experts_to_pause) == (2, 2, 2)
    assert report.registered == report.paused == 0
    backfill.ensure.assert_not_awaited()
    backfill.resume.assert_not_awaited()
    backfill.pause.assert_not_awaited()


@pytest.mark.asyncio
async def test_force_re_registers_every_scope(backfill):
    await backfill_schedules(force=True)

    assert all(
        call.kwargs["force_refresh"] is True
        for call in backfill.ensure.await_args_list + backfill.resume.await_args_list
    )


@pytest.mark.asyncio
async def test_outcomes_are_counted_per_scope(backfill):
    backfill.ensure.side_effect = [FLAGS_OFF, FAILED]
    backfill.pause.return_value = False

    report = await backfill_schedules()

    assert (report.skipped, report.failed, report.already_scheduled) == (1, 3, 1)
    assert report.registered == report.paused == 0


@pytest.mark.asyncio
async def test_rerunning_is_safe(backfill):
    """A second run makes the same calls; what the registry already holds
    comes back as already scheduled."""
    first = await backfill_schedules()
    backfill.ensure.return_value = ALREADY

    second = await backfill_schedules()

    assert first.registered == 2
    assert second.already_scheduled == 3
    assert backfill.ensure.await_count == 4


def test_classify_prefers_registration_then_failure():
    assert registry_backfill._classify(REGISTERED) == "registered"
    assert registry_backfill._classify(FAILED) == "failed"
    assert registry_backfill._classify(ALREADY) == "already_scheduled"
    assert registry_backfill._classify(FLAGS_OFF) == "skipped"
    partly = {
        "community_rebuild": None,
        "dream_nightly_batch": FLAGS_OFF["dream_nightly_batch"],
    }
    assert registry_backfill._classify(partly) == "already_scheduled"
