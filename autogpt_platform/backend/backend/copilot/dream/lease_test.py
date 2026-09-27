"""A sync pass's lease through the real entry point, with the real dream lock
over the in-memory Redis and the pass's row in the in-memory store: written
with the row, renewed before every phase and before apply, emptied when the
row closes; a pass whose lock is no longer its own stops at its next step,
and one that cannot renew before apply does not apply. Only the models, the
budget and apply are stubbed. Every transition that closes a row drops the
lease and the input bundle."""

import logging
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
from pydantic import ValidationError

from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.inference.complete import StructuredCompletion
from backend.copilot.inference.context import InferenceUsage, RouteDecision
from backend.data.dream_pass_models import (
    CLOSED_ROW_CLEARS,
    CLOSED_STATUSES,
    OPEN_STATUSES,
    DreamPassDraft,
    DreamPassUpdate,
)
from backend.data.dream_pass_update import TRANSITION_SQL, transition_args

from . import locks as locks_mod
from . import orchestrator as orchestrator_mod
from .fetch import DreamInput, EpisodeRow
from .lease import renew_sync_lease
from .locks import DEFAULT_LOCK_TTL_SECONDS
from .pass_record import cancelled, cleanup_finished, expired, failed, outcome, reaped
from .pass_run import DreamPassRun, PassEnded
from .schemas import (
    ConsolidationOutput,
    DreamOperations,
    DreamPassResult,
    RecombinationOutput,
)
from .store import write_stop

_SCOPE = MemoryScope.for_user("u")
_LOCK_KEY = _SCOPE.redis_key("dream_lock")
_LONG_AGO = datetime(2000, 1, 1, tzinfo=timezone.utc)
_BILLED = InferenceUsage(
    model="m", input_tokens=100, output_tokens=20, payer="platform_allowance"
)
_ANSWERS = (
    ConsolidationOutput(facts=[]),
    RecombinationOutput(proposals=[]),
    DreamOperations(summary_for_user="ok"),
)


@pytest.fixture(autouse=True)
def sync_pass(mocker) -> None:
    """A sync pass whose route, budget, charges and input are stubbed; the
    lock and the record run for real over the in-memory stand-ins."""
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
    mocker.patch.object(
        orchestrator_mod, "check_dream_budget", AsyncMock(return_value=(True, None))
    )
    mocker.patch.object(
        orchestrator_mod, "gather_dream_input", AsyncMock(return_value=_input())
    )


@pytest.fixture
def apply(mocker, fake_dream_redis) -> AsyncMock:
    """Apply, noting the TTL the pass's lock had when it started."""
    ttls: list[int | None] = []

    async def applied(*_args, **_kwargs):
        ttls.append(fake_dream_redis.ttls.get(_LOCK_KEY))
        return {"session_id": "s", "consolidated_count": 0}

    mock = mocker.patch.object(
        orchestrator_mod, "apply_operations", AsyncMock(side_effect=applied)
    )
    mock.lock_ttls = ttls
    return mock


class TestTheSyncLease:
    async def test_the_row_holds_the_locks_lease_from_insert_to_close(
        self, mocker, fake_dream_db, fake_dream_redis, apply
    ):
        """Before each phase the test lets the lock run nearly out and the
        row's lease lapse; the renewal at the next step restores both."""
        seen: list[tuple[str | None, int | None, datetime | None]] = []

        async def during(call: int) -> None:
            [pass_id] = list(fake_dream_db.rows)
            row = fake_dream_db.rows[pass_id]
            lock = fake_dream_redis.store.get(_LOCK_KEY)
            seen.append(
                (lock, fake_dream_redis.ttls.get(_LOCK_KEY), row["lease_expires_at"])
            )
            fake_dream_redis.ttls[_LOCK_KEY] = 1
            row["lease_expires_at"] = _LONG_AGO

        _phases(mocker, during)
        started = datetime.now(timezone.utc)

        result = await orchestrator_mod.execute_dream_pass("u")

        assert result.error is None and result.skipped is False
        _, draft = fake_dream_db.writes[0]
        assert isinstance(draft, DreamPassDraft) and draft.lease_token
        assert result.started_at is not None
        assert draft.lease_expires_at == result.started_at + timedelta(
            seconds=DEFAULT_LOCK_TTL_SECONDS
        )
        assert [lock for lock, _, _ in seen] == [draft.lease_token] * 3
        assert [ttl for _, ttl, _ in seen] == [DEFAULT_LOCK_TTL_SECONDS] * 3
        renewed_until = started + timedelta(seconds=DEFAULT_LOCK_TTL_SECONDS)
        assert all(lease and lease >= renewed_until for _, _, lease in seen)
        assert apply.lock_ttls == [DEFAULT_LOCK_TTL_SECONDS]
        renewals = _lease_writes(fake_dream_db, result.pass_id)
        assert [update.lease_token for update in renewals] == [draft.lease_token] * 4
        row = fake_dream_db.rows[result.pass_id]
        assert row["status"] is DreamPassStatus.COMPLETE
        assert (row["lease_token"], row["lease_expires_at"]) == (None, None)

    async def test_a_pass_whose_lock_was_taken_stops_before_its_next_phase(
        self, mocker, fake_dream_db, fake_dream_redis, apply
    ):
        """A newer pass took the lock while consolidate ran: the renewal
        before recombine finds it, and the pass stops there with what it was
        billed, leaving the newer pass's lock alone."""

        async def during(call: int) -> None:
            if call == 0:
                fake_dream_redis.store[_LOCK_KEY] = "newer-token"

        phases = _phases(mocker, during)

        result = await orchestrator_mod.execute_dream_pass("u")

        assert result.error == "recombine: dream lock lost before recombine"
        assert phases.await_count == 1
        assert result.usage is not None
        assert [p.phase for p in result.usage.phases] == ["consolidate"]
        apply.assert_not_awaited()
        assert fake_dream_redis.store[_LOCK_KEY] == "newer-token"
        row = fake_dream_db.rows[result.pass_id]
        assert (row["status"], row["error"]) == (DreamPassStatus.ERRORED, result.error)

    async def test_a_renewal_redis_cannot_answer_goes_on_until_apply(
        self, mocker, fake_dream_db, apply, caplog
    ):
        """Unsure is not lost before a phase: the pass goes on through all
        three. Before apply it is not ownership either: the pass ends there,
        with its phases' usage, and applies nothing."""
        mocker.patch.object(
            locks_mod.DreamLockHandle,
            "extend",
            AsyncMock(side_effect=ConnectionError("redis down")),
        )
        phases = _phases(mocker)

        with caplog.at_level(logging.WARNING):
            result = await orchestrator_mod.execute_dream_pass("u")

        assert result.error == "apply: dream lease could not be renewed"
        assert phases.await_count == 3
        assert result.usage is not None and len(result.usage.phases) == 3
        apply.assert_not_awaited()
        assert caplog.text.count("could not renew its lease") == 4
        assert _lease_writes(fake_dream_db, result.pass_id) == []
        row = fake_dream_db.rows[result.pass_id]
        assert (row["status"], row["error"]) == (DreamPassStatus.ERRORED, result.error)

    async def test_a_lease_the_row_will_not_take_does_not_stop_the_pass(
        self, mocker, fake_dream_db, fake_dream_redis, apply, caplog
    ):
        write = fake_dream_db.update_dream_pass

        async def refuses_leases(pass_id: str, update: DreamPassUpdate) -> bool:
            if update.lease_expires_at is not None:
                raise ConnectionError("dream pass database unreachable")
            return await write(pass_id, update)

        mocker.patch.object(fake_dream_db, "update_dream_pass", refuses_leases)
        _phases(mocker)

        with caplog.at_level(logging.WARNING):
            result = await orchestrator_mod.execute_dream_pass("u")

        assert result.error is None
        apply.assert_awaited_once()
        assert caplog.text.count("could not record the lease") == 4
        assert fake_dream_db.rows[result.pass_id]["status"] is DreamPassStatus.COMPLETE

    async def test_a_lost_lock_on_a_stopped_row_ends_with_the_stop(
        self, fake_dream_db, fake_dream_redis
    ):
        """A newer pass expired the row and took the lock between the pass's
        stop check and its renewal: the pass reports the expiry that closed
        its row, not just the lock it lost."""
        run = DreamPassRun.begin("u", "sync_baseline")
        fake_dream_db.seed(_draft(run.pass_id))
        assert await write_stop(run.pass_id, expired("stale", not_updated_since=None))
        async with locks_mod.dream_lock(_SCOPE, token=run.lease_token) as handle:
            run.hold(handle)
            fake_dream_redis.store[_LOCK_KEY] = "newer-token"
            with pytest.raises(PassEnded) as ended:
                await renew_sync_lease(run, "apply")

        assert ended.value.result.error == "expired: stale"
        assert fake_dream_redis.store[_LOCK_KEY] == "newer-token"


class TestAClosingRow:
    @pytest.mark.parametrize(
        "close",
        [
            lambda: outcome(_result(skipped=True), None),
            lambda: outcome(_result(), None),
            lambda: failed("boom", None, datetime.now(timezone.utc)),
            lambda: cancelled("testing", owner_user_id="u"),
            lambda: expired("stale", not_updated_since=None),
        ],
        ids=["skipped", "complete", "errored", "cancelled", "expired"],
    )
    async def test_drops_its_lease_and_bundle_and_keeps_its_outputs(
        self, fake_dream_db, close: Callable[[], DreamPassUpdate]
    ):
        closing = close()
        fake_dream_db.seed(
            _draft("p1"),
            lease_token="tok",
            lease_expires_at=datetime.now(timezone.utc),
            input_bundle=_input(),
            phase_outputs={"consolidate": ConsolidationOutput()},
        )

        assert closing.clear == CLOSED_ROW_CLEARS
        assert await fake_dream_db.update_dream_pass("p1", closing)

        row = fake_dream_db.rows["p1"]
        assert [row[column] for column in sorted(CLOSED_ROW_CLEARS)] == [None] * 3
        assert row["phase_outputs"] == {"consolidate": ConsolidationOutput()}

    def test_a_column_is_set_or_cleared_not_both(self):
        with pytest.raises(ValidationError, match="set and cleared at once"):
            DreamPassUpdate(lease_token="tok", clear=frozenset({"lease_token"}))

    def test_the_statement_empties_the_columns_the_update_clears(self):
        args = transition_args("p1", expired("stale", not_updated_since=None))

        assert args[21] == ["inputBundle", "leaseExpiresAt", "leaseToken"]
        assert transition_args("p1", DreamPassUpdate())[21] == []
        for column in (
            "leaseToken",
            "leaseExpiresAt",
            "inputBundle",
            "cleanupPendingAt",
        ):
            assert f"'{column}' = ANY($22::text[]) THEN NULL" in TRANSITION_SQL

    async def test_the_reapers_close_marks_it_and_only_the_mark_is_written_after(
        self, fake_dream_db
    ):
        """The reaper's expiry keeps the token its cleanup may need and marks
        the row; the write that says the cleanup finished is the one a closed
        row takes, and it only clears."""
        now = datetime.now(timezone.utc)
        fake_dream_db.seed(
            _draft("p1"), lease_token="tok", lease_expires_at=now, updated_at=now
        )

        assert await fake_dream_db.update_dream_pass(
            "p1", reaped("lapsed", not_updated_since=now)
        )
        row = fake_dream_db.rows["p1"]
        assert (row["status"], row["lease_token"]) == (DreamPassStatus.EXPIRED, "tok")
        assert row["cleanup_pending_at"] is not None
        assert not await fake_dream_db.update_dream_pass(
            "p1", failed("late", None, now)
        )

        assert await fake_dream_db.update_dream_pass("p1", cleanup_finished())
        assert (row["cleanup_pending_at"], row["lease_token"]) == (None, None)
        assert (row["status"], row["error"]) == (DreamPassStatus.EXPIRED, "lapsed")

    def test_the_cleanup_write_goes_to_closed_rows_and_only_clears(self):
        with pytest.raises(ValidationError, match="a closed row only has columns"):
            DreamPassUpdate(closed_row=True, error="late")
        closed = transition_args("p1", cleanup_finished())
        opened = transition_args("p1", DreamPassUpdate())

        assert set(closed[17]) == {s.value for s in CLOSED_STATUSES}
        assert set(opened[17]) == {s.value for s in OPEN_STATUSES}
        assert closed[21] == ["cleanupPendingAt", "leaseToken"]
        marked = transition_args(
            "p1", reaped("x", not_updated_since=datetime.now(timezone.utc))
        )
        assert marked[22] is not None and "leaseToken" not in marked[21]


def _phases(
    mocker, during: Callable[[int], Awaitable[None]] | None = None
) -> AsyncMock:
    """Each phase answers, billed; *during* runs with the phase's index while
    it is in flight."""
    calls = 0

    async def answer(*_args, **_kwargs) -> StructuredCompletion:
        nonlocal calls
        if during is not None:
            await during(calls)
        calls += 1
        return StructuredCompletion(value=_ANSWERS[calls - 1], usage=_BILLED)

    return mocker.patch.object(
        orchestrator_mod, "structured_complete", AsyncMock(side_effect=answer)
    )


def _lease_writes(fake_dream_db, pass_id: str) -> list[DreamPassUpdate]:
    """The renewals the pass wrote to its row, in order."""
    return [
        update
        for written, update in fake_dream_db.writes
        if written == pass_id
        and isinstance(update, DreamPassUpdate)
        and update.lease_expires_at is not None
    ]


def _draft(pass_id: str) -> DreamPassDraft:
    return DreamPassDraft(
        id=pass_id,
        user_id="u",
        scope_key=_SCOPE.scope_key,
        route=DreamPassRoute.SYNC,
        trigger=DreamPassTrigger.CRON,
        phase=DreamPassPhase.RECOMBINE,
    )


def _result(*, skipped: bool = False) -> DreamPassResult:
    return DreamPassResult(
        user_id="u",
        pass_id="p1",
        execution_path="sync_baseline",
        skipped=skipped,
        skip_reason="no_input" if skipped else None,
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
