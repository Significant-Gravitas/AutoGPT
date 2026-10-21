"""The guard and the stop path through the sync entry point, with the real
dream lock over the in-memory Redis and the pass's row in the in-memory
store: a pass that finds its scope taken, the admin trigger that forces past
it, the master flag turned off after the cron fired, and a pass cancelled at
each of its checks."""

from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)

from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.inference.complete import StructuredCompletion
from backend.copilot.inference.context import InferenceUsage, RouteDecision
from backend.data.dream_pass_models import DreamPassDraft

from . import orchestrator as orchestrator_mod
from .cancel import cancel_dream_pass
from .fetch import DreamInput, EpisodeRow
from .locks import DEFAULT_LOCK_TTL_SECONDS
from .pass_record import expired
from .schemas import ConsolidationOutput, DreamOperations, RecombinationOutput
from .store import write_stop

_SCOPE = MemoryScope.for_user("u")
_LOCK_KEY = _SCOPE.redis_key("dream_lock")
_BILLED = InferenceUsage(
    model="m",
    input_tokens=100,
    output_tokens=20,
    cost_usd=0.001,
    cost_source="provider",
    payer="platform_allowance",
)
_ANSWERS = (
    ConsolidationOutput(facts=[]),
    RecombinationOutput(proposals=[]),
    DreamOperations(summary_for_user="ok"),
)


def _route(scope, job, *, config=None) -> RouteDecision:
    return RouteDecision(
        engine="provider_sync",
        auth_provider="platform",
        provider="open_router",
        model=f"{job.tier}-model",
        payer="platform_allowance",
        execution_path="sync_baseline",
        cost_log_provider="open_router",
        reason="test",
    )


def _input() -> DreamInput:
    now = datetime.now(timezone.utc)
    episode = EpisodeRow(
        uuid="e1",
        name=None,
        content="hello",
        source_description=None,
        valid_at=None,
        created_at=None,
    )
    return DreamInput(
        user_id="u",
        group_id=_SCOPE.group_id,
        window_start=now - timedelta(days=14),
        window_end=now,
        episodes=[episode],
    )


@pytest.fixture(autouse=True)
def budget(mocker) -> AsyncMock:
    """A sync pass whose budget (returned), route and charges are stubbed;
    the lock, the marker and the record run for real over the in-memory
    stand-ins."""
    mocker.patch.object(orchestrator_mod, "resolve_route", side_effect=_route)
    mocker.patch.object(
        orchestrator_mod, "resolve_dream_execution_path", return_value="sync_baseline"
    )
    mocker.patch.object(
        orchestrator_mod, "is_feature_enabled", AsyncMock(return_value=False)
    )
    mocker.patch.object(
        orchestrator_mod,
        "record_phase_cost",
        AsyncMock(side_effect=lambda ctx, usage: usage),
    )
    return mocker.patch.object(
        orchestrator_mod, "check_dream_budget", AsyncMock(return_value=(True, None))
    )


@pytest.fixture
def gather(mocker) -> AsyncMock:
    return mocker.patch.object(
        orchestrator_mod, "gather_dream_input", AsyncMock(return_value=_input())
    )


@pytest.fixture
def apply(mocker) -> AsyncMock:
    return mocker.patch.object(
        orchestrator_mod,
        "apply_operations",
        AsyncMock(return_value={"session_id": "s", "consolidated_count": 0}),
    )


def _phases(mocker, during: int | None = None, stop=None) -> AsyncMock:
    """Each phase answers, billed; *stop* runs while call *during* is in
    flight, as a cancel landing mid-phase would."""
    calls = 0

    async def answer(*_args, **_kwargs) -> StructuredCompletion:
        nonlocal calls
        if calls == during and stop is not None:
            await stop()
        calls += 1
        return StructuredCompletion(value=_ANSWERS[calls - 1], usage=_BILLED)

    return mocker.patch.object(
        orchestrator_mod, "structured_complete", AsyncMock(side_effect=answer)
    )


def _seed_other_pass(fake_dream_db, **columns) -> str:
    """Another pass of the scope, open at recombine, written a minute ago."""
    draft = DreamPassDraft(
        id="other",
        user_id="u",
        scope_key=_SCOPE.scope_key,
        route=DreamPassRoute.SYNC,
        trigger=DreamPassTrigger.CRON,
        phase=DreamPassPhase.RECOMBINE,
    )
    written = datetime.now(timezone.utc) - timedelta(minutes=1)
    fake_dream_db.seed(draft, **{"updated_at": written, **columns})
    return draft.id


def _cancel_own_pass(fake_dream_db) -> Callable[[], Awaitable[None]]:
    async def cancel() -> None:
        [pass_id] = list(fake_dream_db.rows)
        result = await cancel_dream_pass(pass_id, user_id="u", reason="testing")
        assert result.cancelled

    return cancel


class TestTheGuard:
    @pytest.mark.parametrize("trigger", ["cron", "admin", "eval"])
    async def test_a_pass_behind_a_fresh_open_pass_is_recorded_skipped(
        self, fake_dream_db, fake_dream_redis, gather, trigger
    ):
        other = _seed_other_pass(fake_dream_db)

        result = await orchestrator_mod.execute_dream_pass(
            "u", status_id="job-1" if trigger == "admin" else None, trigger=trigger
        )

        assert (result.skipped, result.skip_reason) == (True, "pass_in_progress")
        gather.assert_not_awaited()
        row = fake_dream_db.rows[result.pass_id]
        assert (row["status"], row["skip_reason"]) == (
            DreamPassStatus.SKIPPED,
            "pass_in_progress",
        )
        assert fake_dream_db.rows[other]["status"] is DreamPassStatus.RUNNING
        assert fake_dream_db.rows[other]["cancel_generation"] == 0
        assert _LOCK_KEY not in fake_dream_redis.store

    async def test_the_admin_trigger_with_force_expires_it_and_runs(
        self, mocker, fake_dream_db, gather, apply
    ):
        other = _seed_other_pass(fake_dream_db)
        _phases(mocker)

        result = await orchestrator_mod.execute_dream_pass(
            "u", status_id="job-1", trigger="admin", force=True
        )

        assert result.error is None and result.skipped is False
        apply.assert_awaited_once()
        row = fake_dream_db.rows[other]
        assert row["status"] is DreamPassStatus.EXPIRED
        assert row["cancel_generation"] == 1
        assert row["error"] == (
            f"forced by an admin-triggered dream pass {result.pass_id}"
        )
        assert fake_dream_db.rows[result.pass_id]["status"] is (
            DreamPassStatus.COMPLETE
        )

    async def test_a_stale_open_pass_is_expired_and_the_pass_runs(
        self, mocker, fake_dream_db, gather, apply
    ):
        stale_at = datetime.now(timezone.utc) - timedelta(
            seconds=DEFAULT_LOCK_TTL_SECONDS + 60
        )
        other = _seed_other_pass(fake_dream_db, updated_at=stale_at)
        _phases(mocker)

        result = await orchestrator_mod.execute_dream_pass("u")

        assert result.error is None and result.skipped is False
        apply.assert_awaited_once()
        assert fake_dream_db.rows[other]["status"] is DreamPassStatus.EXPIRED

    async def test_a_flag_turned_off_after_the_cron_fired_skips_as_disabled(
        self, fake_dream_db, fake_dream_redis, gather, budget, dream_pass_flag
    ):
        dream_pass_flag.return_value = (False, True)

        result = await orchestrator_mod.execute_dream_pass("u")

        assert (result.skipped, result.skip_reason) == (True, "disabled")
        gather.assert_not_awaited()
        budget.assert_not_awaited()
        row = fake_dream_db.rows[result.pass_id]
        assert (row["status"], row["skip_reason"]) == (
            DreamPassStatus.SKIPPED,
            "disabled",
        )
        assert _LOCK_KEY not in fake_dream_redis.store


class TestTheStopChecks:
    async def test_a_pass_reads_its_row_before_each_phase_and_before_apply(
        self, mocker, fake_dream_db, gather, apply
    ):
        _phases(mocker)
        reads = mocker.spy(fake_dream_db, "get_dream_pass")

        result = await orchestrator_mod.execute_dream_pass("u")

        assert result.error is None
        assert [call.args for call in reads.call_args_list] == [(result.pass_id,)] * 4

    @pytest.mark.parametrize(
        "during, billed",
        [
            (None, []),
            (0, ["consolidate"]),
            (1, ["consolidate", "recombine"]),
            (2, ["consolidate", "recombine", "sanitize"]),
        ],
        ids=["before_consolidate", "before_recombine", "before_sanitize", "pre_apply"],
    )
    async def test_a_cancelled_pass_stops_at_its_next_check(
        self, mocker, fake_dream_db, fake_dream_redis, gather, apply, during, billed
    ):
        cancel = _cancel_own_pass(fake_dream_db)
        if during is None:
            gather.side_effect = _gather_then(cancel)
        phases = _phases(mocker, during, cancel)

        result = await orchestrator_mod.execute_dream_pass("u")

        assert result.error == "cancelled: testing"
        assert result.usage is not None
        assert [p.phase for p in result.usage.phases] == billed
        assert phases.await_count == len(billed)
        apply.assert_not_awaited()
        assert _LOCK_KEY not in fake_dream_redis.store
        assert _SCOPE.redis_key("last_completed") not in fake_dream_redis.store
        row = fake_dream_db.rows[result.pass_id]
        assert (row["status"], row["error"], row["cancel_generation"]) == (
            DreamPassStatus.CANCELLED,
            "testing",
            1,
        )
        assert DreamPassStatus.ERRORED not in fake_dream_db.statuses(result.pass_id)

    async def test_a_pass_a_newer_one_expired_stops_the_same_way(
        self, mocker, fake_dream_db, gather, apply
    ):
        async def expire() -> None:
            [pass_id] = list(fake_dream_db.rows)
            await write_stop(pass_id, expired("forced", not_updated_since=None))

        _phases(mocker, 1, expire)

        result = await orchestrator_mod.execute_dream_pass("u")

        assert result.error == "expired: forced"
        assert result.usage is not None
        assert [p.phase for p in result.usage.phases] == ["consolidate", "recombine"]
        apply.assert_not_awaited()
        assert fake_dream_db.rows[result.pass_id]["status"] is DreamPassStatus.EXPIRED


def _gather_then(
    stop: Callable[[], Awaitable[None]],
) -> Callable[[MemoryScope], Awaitable[DreamInput]]:
    """A gather during which *stop* lands."""

    async def gather(scope: MemoryScope) -> DreamInput:
        await stop()
        return _input()

    return gather
