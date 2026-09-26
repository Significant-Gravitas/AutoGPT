"""The dream store: each transition the two routes write, the read side, and
that a failed write never reaches the pass."""

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.dream_pass_models import (
    DreamPassApplied,
    DreamPassDraft,
    DreamPassOperations,
    DreamPassRecord,
    DreamPassUpdate,
    DreamPhaseOutputs,
)

from . import store
from .fetch import DreamInput
from .pass_record import dream_pass_result_from_row
from .schemas import (
    ConsolidationOutput,
    DreamOperations,
    DreamOperationsSnapshot,
    DreamPassResult,
    DreamPassUsage,
    IngestionDrainStatus,
    PhaseUsage,
    RecombinationOutput,
    WriteSummary,
)

_STARTED = datetime(2026, 9, 26, 3, 0, tzinfo=timezone.utc)
_FINISHED = _STARTED + timedelta(minutes=7)


@pytest.fixture
def db(mocker) -> MagicMock:
    """The accessor's target, with every write succeeding."""
    database = MagicMock()
    database.create_dream_pass = AsyncMock()
    database.update_dream_pass = AsyncMock(return_value=True)
    database.get_dream_pass_for_user = AsyncMock()
    mocker.patch.object(store, "dream_db", return_value=database)
    return database


def _update(db: MagicMock) -> tuple[str, DreamPassUpdate]:
    db.update_dream_pass.assert_awaited_once()
    pass_id, update = db.update_dream_pass.await_args.args
    return pass_id, update


def _usage(cost: float | None = 0.03) -> DreamPassUsage:
    return DreamPassUsage(
        phases=[
            PhaseUsage(
                phase="consolidate",
                model="claude-sonnet-5",
                input_tokens=100,
                output_tokens=40,
                cost_usd=cost,
            )
        ],
        total_input_tokens=100,
        total_output_tokens=40,
        total_cost_usd=cost,
        discount_applied=0.5,
    )


def _bundle() -> DreamInput:
    return DreamInput(
        user_id="u1",
        group_id="user_u1",
        window_start=_STARTED - timedelta(days=14),
        window_end=_STARTED,
    )


def _complete_result(**overrides) -> DreamPassResult:
    fields = {
        "user_id": "u1",
        "pass_id": "p1",
        "started_at": _STARTED,
        "completed_at": _FINISHED,
        "elapsed_seconds": 420.0,
        "consolidated_count": 3,
        "proposal_count": 1,
        "demotion_count": 2,
        "entity_invalidation_count": 0,
        "summary_for_user": "Tidied what you told me this week.",
        "dream_session_id": "session-1",
        "ingestion_drain_status": IngestionDrainStatus.drained,
        "operations": DreamOperationsSnapshot(
            writes=[WriteSummary(content="Nick ships on Fridays")]
        ),
        "usage": _usage(),
    }
    return DreamPassResult(**{**fields, **overrides})


class TestWrites:
    @pytest.mark.parametrize(
        "route, trigger, expected_route, expected_trigger",
        [
            ("sync_baseline", "cron", DreamPassRoute.SYNC, DreamPassTrigger.CRON),
            (
                "anthropic_batch",
                "admin",
                DreamPassRoute.ANTHROPIC_BATCH,
                DreamPassTrigger.ADMIN,
            ),
            ("sync_baseline", "eval", DreamPassRoute.SYNC, DreamPassTrigger.EVAL),
        ],
    )
    async def test_start_inserts_a_running_row_gathering(
        self, db, route, trigger, expected_route, expected_trigger
    ):
        scope = MemoryScope.for_expert("u1", "expert-1")

        await store.start_pass(
            "p1", scope, route=route, trigger=trigger, started_at=_STARTED
        )

        db.create_dream_pass.assert_awaited_once_with(
            DreamPassDraft(
                id="p1",
                user_id="u1",
                expert_id="expert-1",
                scope_key=scope.scope_key,
                route=expected_route,
                trigger=expected_trigger,
                status=DreamPassStatus.RUNNING,
                phase=DreamPassPhase.GATHER,
                started_at=_STARTED,
            )
        )

    async def test_gathered_records_the_window_and_moves_to_consolidate(self, db):
        bundle = _bundle()

        await store.record_gathered("p1", bundle)

        assert _update(db) == (
            "p1",
            DreamPassUpdate(
                phase=DreamPassPhase.CONSOLIDATE,
                window_start=bundle.window_start,
                window_end=bundle.window_end,
            ),
        )

    @pytest.mark.parametrize(
        "phase, output, next_step",
        [
            ("consolidate", ConsolidationOutput(), DreamPassPhase.RECOMBINE),
            ("recombine", RecombinationOutput(), DreamPassPhase.SANITIZE),
            ("sanitize", DreamOperations(summary_for_user="ok"), DreamPassPhase.APPLY),
        ],
    )
    async def test_phase_output_is_kept_and_the_pass_moves_on(
        self, db, phase, output, next_step
    ):
        await store.record_phase_output("p1", phase, output)

        _, update = _update(db)
        assert update.phase is next_step
        assert update.status is None
        assert update.phase_outputs == DreamPhaseOutputs.model_validate({phase: output})

    async def test_applying_keeps_the_clamped_operations(self, db):
        ops = DreamOperations(summary_for_user="clamped")

        await store.record_applying("p1", ops)

        assert _update(db) == (
            "p1",
            DreamPassUpdate(
                status=DreamPassStatus.APPLYING,
                phase=DreamPassPhase.APPLY,
                operations=DreamPassOperations(planned=ops),
            ),
        )

    async def test_submitted_records_the_batch_the_bundle_and_the_lease(self, db):
        bundle = _bundle()
        before = datetime.now(timezone.utc)

        await store.record_submitted(
            "p1",
            input_bundle=bundle,
            provider_batch_id="msgbatch_1",
            lease_token="tok",
            lease_ttl_seconds=3600,
        )

        _, update = _update(db)
        assert update.status is DreamPassStatus.SUBMITTED
        assert update.phase is DreamPassPhase.CONSOLIDATE
        assert update.provider_batch_id == "msgbatch_1"
        assert update.input_bundle == bundle
        assert update.lease_token == "tok"
        assert update.submitted_at is not None and update.submitted_at >= before
        assert update.lease_expires_at == update.submitted_at + timedelta(hours=1)

    async def test_next_batch_names_the_phase_it_runs(self, db):
        await store.record_next_batch("p1", "recombine", "msgbatch_2")

        assert _update(db) == (
            "p1",
            DreamPassUpdate(
                phase=DreamPassPhase.RECOMBINE, provider_batch_id="msgbatch_2"
            ),
        )

    async def test_a_completed_sync_pass_records_what_apply_reported(self, db):
        result = _complete_result()

        await store.record_sync_outcome(result)

        _, update = _update(db)
        assert update.status is DreamPassStatus.COMPLETE
        assert update.phase is DreamPassPhase.DONE
        assert update.usage == result.usage
        assert update.applied_at == update.completed_at == _FINISHED
        assert update.operations == DreamPassOperations(
            applied=DreamPassApplied(
                consolidated_count=3,
                proposal_count=1,
                demotion_count=2,
                summary_for_user="Tidied what you told me this week.",
                dream_session_id="session-1",
                ingestion_drain_status=IngestionDrainStatus.drained,
                snapshot=result.operations,
            )
        )

    async def test_a_skipped_pass_records_its_reason(self, db):
        result = DreamPassResult(
            user_id="u1",
            pass_id="p1",
            completed_at=_FINISHED,
            skipped=True,
            skip_reason="no_new_activity",
        )

        await store.record_sync_outcome(result)

        assert _update(db) == (
            "p1",
            DreamPassUpdate(
                status=DreamPassStatus.SKIPPED,
                skip_reason="no_new_activity",
                completed_at=_FINISHED,
            ),
        )

    async def test_a_failed_pass_keeps_its_partial_usage_and_a_capped_error(self, db):
        result = DreamPassResult(
            user_id="u1",
            pass_id="p1",
            completed_at=_FINISHED,
            error="recombine: " + "x" * 5000,
            usage=_usage(),
        )

        await store.record_sync_outcome(result)

        _, update = _update(db)
        assert update.status is DreamPassStatus.ERRORED
        assert update.error == result.error[:2000]
        assert update.usage == result.usage
        assert update.completed_at == _FINISHED
        assert update.phase is None

    async def test_a_pass_handed_to_the_batch_route_stays_open(self, db):
        await store.record_sync_outcome(
            DreamPassResult(
                user_id="u1", pass_id="p1", execution_path="anthropic_batch"
            )
        )

        db.update_dream_pass.assert_not_awaited()

    async def test_a_batch_route_pass_that_failed_before_submit_is_closed(self, db):
        await store.record_sync_outcome(
            DreamPassResult(
                user_id="u1",
                pass_id="p1",
                execution_path="anthropic_batch",
                error="anthropic_batch: phase 1 submit failed: boom",
            )
        )

        _, update = _update(db)
        assert update.status is DreamPassStatus.ERRORED

    async def test_a_completed_batch_pass_records_the_usage_it_is_given(self, db):
        usage = _usage(cost=0.011)
        result = _complete_result(execution_path="anthropic_batch", usage=None)

        await store.record_batch_complete(result, usage)

        _, update = _update(db)
        assert update.status is DreamPassStatus.COMPLETE
        assert update.usage == usage

    async def test_a_failed_batch_pass_records_the_error_and_usage(self, db):
        usage = _usage()

        await store.record_batch_failed("p1", "recombine: provider down", usage)

        _, update = _update(db)
        assert update.status is DreamPassStatus.ERRORED
        assert update.error == "recombine: provider down"
        assert update.usage == usage
        assert update.completed_at is not None


class TestAFailedWriteNeverFailsThePass:
    @pytest.mark.parametrize(
        "write",
        [
            lambda: store.record_gathered("p1", _bundle()),
            lambda: store.record_phase_output(
                "p1", "consolidate", ConsolidationOutput()
            ),
            lambda: store.record_applying("p1", DreamOperations()),
            lambda: store.record_submitted(
                "p1",
                input_bundle=_bundle(),
                provider_batch_id="b",
                lease_token="t",
                lease_ttl_seconds=60,
            ),
            lambda: store.record_next_batch("p1", "sanitize", "b"),
            lambda: store.record_sync_outcome(_complete_result()),
            lambda: store.record_batch_complete(_complete_result(), None),
            lambda: store.record_batch_failed("p1", "boom", None),
        ],
    )
    async def test_an_update_that_raises_is_logged_at_warning(self, db, caplog, write):
        db.update_dream_pass.side_effect = ConnectionError("db down")

        with caplog.at_level(logging.WARNING, logger=store.logger.name):
            await write()

        assert any(
            r.levelno == logging.WARNING and "p1" in r.getMessage()
            for r in caplog.records
        )

    async def test_an_insert_that_raises_is_logged_at_warning(self, db, caplog):
        db.create_dream_pass.side_effect = ConnectionError("db down")

        with caplog.at_level(logging.WARNING, logger=store.logger.name):
            await store.start_pass(
                "p1",
                MemoryScope.for_user("u1"),
                route="sync_baseline",
                trigger="cron",
                started_at=_STARTED,
            )

        assert "could not insert" in caplog.text

    async def test_an_unreachable_accessor_is_logged_at_warning(self, mocker, caplog):
        mocker.patch.object(store, "dream_db", side_effect=RuntimeError("no rpc"))

        with caplog.at_level(logging.WARNING, logger=store.logger.name):
            await store.record_next_batch("p1", "sanitize", "b")

        assert "could not record the next batch" in caplog.text

    async def test_an_output_that_is_not_the_phase_is_dropped_not_raised(
        self, db, caplog
    ):
        with caplog.at_level(logging.WARNING, logger=store.logger.name):
            await store.record_phase_output("p1", "consolidate", RecombinationOutput())

        db.update_dream_pass.assert_not_awaited()
        assert "could not record the consolidate output" in caplog.text

    async def test_a_row_that_is_missing_or_closed_is_logged(self, db, caplog):
        db.update_dream_pass.return_value = False

        with caplog.at_level(logging.WARNING, logger=store.logger.name):
            await store.record_next_batch("p1", "sanitize", "b")

        assert "no open record" in caplog.text


class TestReadSide:
    async def test_read_is_owner_scoped(self, db):
        row = _row()
        db.get_dream_pass_for_user.return_value = row

        assert await store.read_dream_pass("p1", user_id="u1") is row
        db.get_dream_pass_for_user.assert_awaited_once_with("p1", "u1")

    async def test_a_failed_read_raises(self, db):
        db.get_dream_pass_for_user.side_effect = ConnectionError("db down")

        with pytest.raises(ConnectionError):
            await store.read_dream_pass("p1", user_id="u1")

    def test_a_completed_batch_row_reads_back_with_usage_and_timings(self):
        usage = _usage(cost=0.011)
        snapshot = DreamOperationsSnapshot(writes=[WriteSummary(content="fact")])
        row = _row(
            route=DreamPassRoute.ANTHROPIC_BATCH,
            status=DreamPassStatus.COMPLETE,
            phase=DreamPassPhase.DONE,
            completed_at=_FINISHED,
            usage=usage,
            operations=DreamPassOperations(
                applied=DreamPassApplied(
                    consolidated_count=1,
                    proposal_count=2,
                    summary_for_user="ok",
                    dream_session_id="s1",
                    ingestion_drain_status=IngestionDrainStatus.skipped,
                    snapshot=snapshot,
                )
            ),
        )

        result = dream_pass_result_from_row(row)

        assert result == DreamPassResult(
            user_id="u1",
            pass_id="p1",
            started_at=_STARTED,
            completed_at=_FINISHED,
            elapsed_seconds=420.0,
            execution_path="anthropic_batch",
            consolidated_count=1,
            proposal_count=2,
            summary_for_user="ok",
            dream_session_id="s1",
            ingestion_drain_status=IngestionDrainStatus.skipped,
            operations=snapshot,
            usage=usage,
        )

    def test_a_skipped_row_reads_as_skipped(self):
        row = _row(
            status=DreamPassStatus.SKIPPED,
            skip_reason="lock_held",
            completed_at=_FINISHED,
        )

        result = dream_pass_result_from_row(row)

        assert result.skipped is True
        assert result.skip_reason == "lock_held"
        assert result.operations is None
        assert result.usage is None

    def test_an_errored_row_reads_with_its_error_and_usage(self):
        usage = _usage()
        row = _row(
            status=DreamPassStatus.ERRORED,
            error="sanitize: bad json",
            usage=usage,
            completed_at=_FINISHED,
        )

        result = dream_pass_result_from_row(row)

        assert (result.error, result.usage, result.skipped) == (
            "sanitize: bad json",
            usage,
            False,
        )

    def test_a_pass_in_flight_has_no_end_yet(self):
        row = _row(
            route=DreamPassRoute.ANTHROPIC_BATCH,
            status=DreamPassStatus.SUBMITTED,
            phase=DreamPassPhase.RECOMBINE,
        )

        result = dream_pass_result_from_row(row)

        assert result.completed_at is None
        assert result.elapsed_seconds is None
        assert result.usage is None
        assert result.error is None and result.skipped is False

    async def test_a_sync_outcome_reads_back_as_the_result_it_recorded(self, db):
        """What the orchestrator returned is what the eval driver reads."""
        result = _complete_result()
        await store.record_sync_outcome(result)
        _, update = _update(db)
        assert update.operations is not None

        row = _row(
            status=update.status,
            phase=update.phase,
            operations=update.operations,
            usage=update.usage,
            completed_at=update.completed_at,
        )

        assert dream_pass_result_from_row(row) == result


def _row(**overrides) -> DreamPassRecord:
    fields = {
        "id": "p1",
        "user_id": "u1",
        "expert_id": None,
        "scope_key": "u1",
        "route": DreamPassRoute.SYNC,
        "trigger": DreamPassTrigger.CRON,
        "phase": DreamPassPhase.GATHER,
        "status": DreamPassStatus.RUNNING,
        "skip_reason": None,
        "cancel_generation": 0,
        "provider_batch_id": None,
        "lease_token": None,
        "lease_expires_at": None,
        "input_bundle": None,
        "phase_outputs": DreamPhaseOutputs(),
        "operations": DreamPassOperations(),
        "usage": None,
        "window_start": None,
        "window_end": None,
        "error": None,
        "created_at": _STARTED,
        "started_at": _STARTED,
        "submitted_at": None,
        "applied_at": None,
        "completed_at": None,
        "updated_at": _STARTED,
    }
    return DreamPassRecord(**{**fields, **overrides})
