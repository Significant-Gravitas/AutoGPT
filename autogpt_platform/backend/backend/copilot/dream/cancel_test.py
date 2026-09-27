"""The stop path with the store mocked the way ``store_test.py`` mocks it: the
cancel's one transition and what it reads back, cancelling a scope's open
passes for the wipe, and the checks a running pass makes, which stop it on a
cancelled or expired row and never on a store that does not answer."""

import asyncio
import logging
from datetime import datetime, timezone
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
    DreamPassOperations,
    DreamPassRecord,
    DreamPhaseOutputs,
)

from . import cancel, store
from .batch_outcome import BatchPass
from .cancel import (
    DreamPassCancel,
    cancel_dream_pass,
    cancel_open_passes,
    end_batch_pass_if_stopped,
    stop_if_stopped,
)
from .pass_record import MAX_ERROR_CHARS
from .pass_run import DreamPassRun, PassEnded
from .schemas import PhaseUsage

_NOW = datetime.now(timezone.utc)
_SCOPE = MemoryScope.for_user("u1")
_BATCH_PASS = BatchPass(
    user_id="u1", expert_id=None, pass_id="p1", job_id="j1", phase_models={}
)


@pytest.fixture
def db(mocker) -> MagicMock:
    """The accessor's target, every write landing and no row to read."""
    database = MagicMock()
    database.update_dream_pass = AsyncMock(return_value=True)
    database.get_dream_pass = AsyncMock(return_value=None)
    database.get_dream_pass_for_user = AsyncMock(return_value=None)
    database.list_open_dream_passes = AsyncMock(return_value=[])
    mocker.patch.object(store, "dream_db", return_value=database)
    return database


async def _hang(*_args) -> None:
    await asyncio.Event().wait()


def _row(pass_id: str = "p1", **overrides) -> DreamPassRecord:
    fields = {
        "id": pass_id,
        "user_id": "u1",
        "expert_id": None,
        "scope_key": _SCOPE.scope_key,
        "route": DreamPassRoute.SYNC,
        "trigger": DreamPassTrigger.CRON,
        "phase": DreamPassPhase.RECOMBINE,
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
        "created_at": _NOW,
        "started_at": _NOW,
        "submitted_at": None,
        "applied_at": None,
        "completed_at": None,
        "updated_at": _NOW,
    }
    return DreamPassRecord(**{**fields, **overrides})


def _cancelled(pass_id: str = "p1", **overrides) -> DreamPassRecord:
    return _row(
        pass_id,
        status=DreamPassStatus.CANCELLED,
        cancel_generation=1,
        error="testing",
        completed_at=_NOW,
        **overrides,
    )


class TestCancelDreamPass:
    async def test_an_open_pass_is_closed_cancelled_and_read_back(self, db):
        db.get_dream_pass_for_user.return_value = _cancelled()

        result = await cancel_dream_pass("p1", user_id="u1", reason="testing")

        assert result == DreamPassCancel(cancelled=True, record=_cancelled())
        pass_id, update = db.update_dream_pass.await_args.args
        assert pass_id == "p1"
        assert (update.status, update.error, update.owner_user_id) == (
            DreamPassStatus.CANCELLED,
            "testing",
            "u1",
        )
        assert update.bump_cancel_generation is True
        assert update.completed_at is not None
        assert update.not_updated_since is None
        db.get_dream_pass_for_user.assert_awaited_once_with("p1", "u1")

    async def test_a_pass_that_already_ended_is_left_as_it_was(self, db):
        finished = _row(status=DreamPassStatus.COMPLETE, phase=DreamPassPhase.DONE)
        db.update_dream_pass.return_value = False
        db.get_dream_pass_for_user.return_value = finished

        result = await cancel_dream_pass("p1", user_id="u1", reason="testing")

        assert result == DreamPassCancel(cancelled=False, record=finished)

    async def test_a_missing_or_another_users_pass_reads_as_none(self, db):
        db.update_dream_pass.return_value = False

        result = await cancel_dream_pass("p1", user_id="u2", reason="testing")

        assert result == DreamPassCancel(cancelled=False, record=None)
        _, update = db.update_dream_pass.await_args.args
        assert update.owner_user_id == "u2"

    async def test_a_long_reason_is_capped_like_any_error(self, db):
        await cancel_dream_pass("p1", user_id="u1", reason="x" * (MAX_ERROR_CHARS * 2))

        _, update = db.update_dream_pass.await_args.args
        assert update.error == "x" * MAX_ERROR_CHARS

    @pytest.mark.parametrize(
        "stalled",
        [lambda db: db.update_dream_pass, lambda db: db.get_dream_pass_for_user],
        ids=["the_cancel", "the_read_back"],
    )
    async def test_a_store_that_does_not_answer_raises_at_its_deadline(
        self, db, monkeypatch, stalled
    ):
        monkeypatch.setattr(store, "RECORD_WRITE_TIMEOUT_SECONDS", 0.05)
        stalled(db).side_effect = _hang
        loop = asyncio.get_running_loop()
        started = loop.time()

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                cancel_dream_pass("p1", user_id="u1", reason="testing"), 5
            )

        # The store's deadline, not this test's own guard, gave up.
        assert loop.time() - started < 2

    async def test_a_failed_read_back_raises(self, db):
        db.get_dream_pass_for_user.side_effect = ConnectionError("db down")

        with pytest.raises(ConnectionError):
            await cancel_dream_pass("p1", user_id="u1", reason="testing")


class TestCancelOpenPasses:
    async def test_every_open_pass_of_the_scope_is_cancelled(self, db):
        db.list_open_dream_passes.return_value = [_row("a"), _row("b")]
        db.get_dream_pass_for_user.side_effect = [_cancelled("a"), _cancelled("b")]

        results = await cancel_open_passes(_SCOPE, user_id="u1", reason="wiped")

        assert results == [
            DreamPassCancel(cancelled=True, record=_cancelled("a")),
            DreamPassCancel(cancelled=True, record=_cancelled("b")),
        ]
        # Every open row, however many: the wipe must not leave one running.
        db.list_open_dream_passes.assert_awaited_once_with(_SCOPE.scope_key, limit=None)
        sent = [call.args for call in db.update_dream_pass.await_args_list]
        assert [pass_id for pass_id, _ in sent] == ["a", "b"]
        assert {(u.error, u.owner_user_id) for _, u in sent} == {("wiped", "u1")}

    async def test_a_scope_the_store_cannot_list_raises(self, db):
        db.list_open_dream_passes.side_effect = ConnectionError("db down")

        with pytest.raises(ConnectionError):
            await cancel_open_passes(_SCOPE, user_id="u1", reason="wiped")


class TestTheSyncCheck:
    async def test_a_cancelled_row_ends_the_pass_with_its_usage(self, db):
        run = DreamPassRun.begin("u1", "sync_baseline")
        run.phases.append(PhaseUsage(phase="consolidate", model="m", input_tokens=10))
        db.get_dream_pass.return_value = _cancelled(run.pass_id)

        with pytest.raises(PassEnded) as ended:
            await stop_if_stopped(run)

        result = ended.value.result
        assert result.error == "cancelled: testing"
        assert result.skipped is False
        assert result.usage is not None and result.usage.phases == run.phases
        db.get_dream_pass.assert_awaited_once_with(run.pass_id)

    async def test_an_expired_row_ends_the_pass_too(self, db):
        run = DreamPassRun.begin("u1", "sync_baseline")
        db.get_dream_pass.return_value = _row(
            run.pass_id,
            status=DreamPassStatus.EXPIRED,
            cancel_generation=1,
            error="no progress since then; expired by dream pass p9",
        )

        with pytest.raises(PassEnded) as ended:
            await stop_if_stopped(run)

        assert ended.value.result.error == (
            "expired: no progress since then; expired by dream pass p9"
        )

    @pytest.mark.parametrize(
        "row",
        [_row(), _row(status=DreamPassStatus.APPLYING), None],
        ids=["running", "applying", "no_row"],
    )
    async def test_a_row_nobody_stopped_lets_the_pass_go_on(self, db, row):
        db.get_dream_pass.return_value = row

        await stop_if_stopped(DreamPassRun.begin("u1", "sync_baseline"))

    async def test_a_check_the_store_cannot_answer_lets_the_pass_go_on(
        self, db, monkeypatch, caplog
    ):
        monkeypatch.setattr(store, "RECORD_WRITE_TIMEOUT_SECONDS", 0.05)
        db.get_dream_pass.side_effect = _hang

        with caplog.at_level(logging.WARNING, logger=cancel.logger.name):
            await asyncio.wait_for(
                stop_if_stopped(DreamPassRun.begin("u1", "sync_baseline")), 5
            )

        assert "check for a stop" in caplog.text


class TestTheBatchCheck:
    @pytest.fixture
    def ends(self, mocker) -> tuple[AsyncMock, AsyncMock]:
        provider = mocker.patch.object(cancel, "cancel_provider_batch", AsyncMock())
        fail = mocker.patch.object(cancel, "fail_pass", AsyncMock())
        return provider, fail

    async def test_a_stopped_pass_cancels_its_batch_and_ends_through_fail_pass(
        self, db, ends
    ):
        provider, fail = ends
        db.get_dream_pass.return_value = _cancelled(
            route=DreamPassRoute.ANTHROPIC_BATCH, provider_batch_id="msgbatch_1"
        )

        assert await end_batch_pass_if_stopped(_BATCH_PASS) is True

        provider.assert_awaited_once_with("msgbatch_1")
        fail.assert_awaited_once_with(_BATCH_PASS, "cancelled: testing")

    async def test_a_stopped_row_without_a_batch_skips_the_provider(self, db, ends):
        provider, fail = ends
        db.get_dream_pass.return_value = _cancelled()

        assert await end_batch_pass_if_stopped(_BATCH_PASS) is True

        provider.assert_not_awaited()
        fail.assert_awaited_once()

    async def test_a_pass_nobody_stopped_goes_on(self, db, ends):
        provider, fail = ends
        db.get_dream_pass.return_value = _row(status=DreamPassStatus.SUBMITTED)

        assert await end_batch_pass_if_stopped(_BATCH_PASS) is False

        provider.assert_not_awaited()
        fail.assert_not_awaited()

    async def test_a_check_the_store_cannot_answer_goes_on(self, db, ends):
        provider, fail = ends
        db.get_dream_pass.side_effect = ConnectionError("db down")

        assert await end_batch_pass_if_stopped(_BATCH_PASS) is False

        fail.assert_not_awaited()
