"""The dream batch deliveries that never run the phase chain, over the
in-memory store and Redis: the executor's check that drops a closed pass's
batch before it is polled, a dead-end payload, and a duplicate of the last
phase. The provider and the charges are stubbed at their edges."""

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)
from pydantic import BaseModel

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.dream_pass_models import DreamPassDraft
from backend.executor.batch_executor import PendingEntry
from backend.util.llm.providers import BatchResultRow

from . import job_status
from .batch_callbacks import handle_dream_batch_result
from .batch_deliveries import DUPLICATE_ERROR, should_dispatch
from .batch_state import state_key, write_phase_to_state
from .batch_submit import input_bundle_key, persist_input_bundle
from .cancel import cancel_dream_pass
from .fetch import DreamInput
from .pass_record import expired
from .schemas import (
    ConsolidationOutput,
    DreamOperations,
    DreamPhase,
    RecombinationOutput,
)
from .store import write_stop

_SCOPE = MemoryScope.for_user("u1")
_LOCK_KEY = _SCOPE.redis_key("dream_lock")
_MODELS = {p: "claude-sonnet-5" for p in ("consolidate", "recombine", "sanitize")}


@pytest.fixture(autouse=True)
def provider_cancel(mocker) -> AsyncMock:
    mocker.patch(
        "backend.copilot.dream.provider_batch.anthropic_api_key", return_value="k"
    )
    return mocker.patch(
        "backend.copilot.dream.provider_batch.cancel_batch",
        AsyncMock(return_value=True),
    )


@pytest.fixture(autouse=True)
def charges(mocker) -> AsyncMock:
    return mocker.patch(
        "backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock()
    )


class TestShouldDispatch:
    async def test_a_cancelled_pass_has_its_batch_cancelled_and_is_ended(
        self, fake_dream_db, fake_dream_redis, provider_cancel, charges
    ):
        """Cancelled while recombine's batch was in flight: the executor's
        check cancels that batch, charges the phase that landed, releases the
        lock and cleans the pass up, and says not to poll it."""
        await _in_flight(fake_dream_db, fake_dream_redis)
        assert (await cancel_dream_pass("p1", user_id="u1", reason="testing")).cancelled
        provider_cancel.reset_mock()

        assert await should_dispatch(_entry("recombine")) is False

        provider_cancel.assert_awaited_once()
        assert provider_cancel.await_args.kwargs["provider_batch_id"] == "b-rec"
        assert _charged(charges) == ["consolidate"]
        assert _LOCK_KEY not in fake_dream_redis.store
        assert state_key("p1") not in fake_dream_redis.hashes
        assert input_bundle_key("p1") not in fake_dream_redis.store
        status = await job_status.read_status(kind="dream_pass", job_id="j1")
        assert status is not None
        assert (status.state, status.error) == ("errored", "cancelled: testing")
        # The job keeps what the landed phase used, which the closed row
        # refused.
        assert status.result is not None
        phases = status.result["usage"]["phases"]
        assert [phase["phase"] for phase in phases] == ["consolidate"]
        row = fake_dream_db.rows["p1"]
        assert (row["status"], row["error"]) == (DreamPassStatus.CANCELLED, "testing")
        assert row.get("usage") is None

    async def test_an_expired_pass_whose_lock_a_newer_pass_holds_keeps_that_lock(
        self, fake_dream_db, fake_dream_redis
    ):
        await _in_flight(fake_dream_db, fake_dream_redis)
        assert await write_stop("p1", expired("stale", not_updated_since=None))
        fake_dream_redis.store[_LOCK_KEY] = "newer-token"

        assert await should_dispatch(_entry("recombine")) is False

        assert fake_dream_redis.store[_LOCK_KEY] == "newer-token"
        assert fake_dream_db.rows["p1"]["status"] is DreamPassStatus.EXPIRED

    async def test_an_open_pass_is_dispatched_untouched(
        self, fake_dream_db, fake_dream_redis, provider_cancel, charges
    ):
        await _in_flight(fake_dream_db, fake_dream_redis)

        assert await should_dispatch(_entry("recombine")) is True

        provider_cancel.assert_not_awaited()
        charges.assert_not_awaited()
        assert fake_dream_redis.store[_LOCK_KEY] == "tok"
        assert state_key("p1") in fake_dream_redis.hashes

    async def test_a_row_the_store_cannot_read_is_dispatched_as_before(
        self, fake_dream_db, fake_dream_redis, provider_cancel, caplog
    ):
        await _in_flight(fake_dream_db, fake_dream_redis)
        fake_dream_db.fail = True

        with caplog.at_level(logging.WARNING):
            assert await should_dispatch(_entry("recombine")) is True

        provider_cancel.assert_not_awaited()
        assert "could not read its row before polling its batch" in caplog.text

    async def test_a_stalled_row_read_gives_up_at_the_checks_own_deadline(
        self, monkeypatch, stalled_dream_db, provider_cancel
    ):
        """The executor walks its queue serially, so the check's read gets a
        deadline of its own, shorter than a record write's."""
        monkeypatch.setattr(
            "backend.copilot.dream.store.RECORD_WRITE_TIMEOUT_SECONDS", 30
        )
        monkeypatch.setattr(
            "backend.copilot.dream.batch_deliveries."
            "DISPATCH_CHECK_READ_TIMEOUT_SECONDS",
            0.05,
        )

        dispatched = await asyncio.wait_for(should_dispatch(_entry("recombine")), 5)

        assert dispatched is True
        assert (stalled_dream_db.started, stalled_dream_db.cancelled) == (1, 1)
        provider_cancel.assert_not_awaited()

    @pytest.mark.parametrize("pass_id", ["", "never-inserted"])
    async def test_a_batch_with_no_row_to_read_is_dispatched(
        self, fake_dream_db, provider_cancel, pass_id
    ):
        entry = _entry("recombine")
        entry.payload["pass_id"] = pass_id

        assert await should_dispatch(entry) is True

        provider_cancel.assert_not_awaited()


class TestADeadEnd:
    async def test_charges_the_landed_phases_and_cleans_the_pass_up(
        self, fake_dream_db, fake_dream_redis, charges
    ):
        await _in_flight(fake_dream_db, fake_dream_redis)
        entry = _entry("recombine")
        entry.payload["phase"] = "daydream"

        await handle_dream_batch_result(entry, [])

        row = fake_dream_db.rows["p1"]
        assert (row["status"], row["error"]) == (
            DreamPassStatus.ERRORED,
            "unknown batch phase 'daydream'",
        )
        assert _charged(charges) == ["consolidate"]
        assert state_key("p1") not in fake_dream_redis.hashes
        assert input_bundle_key("p1") not in fake_dream_redis.store
        assert _LOCK_KEY not in fake_dream_redis.store

    async def test_without_an_owner_closes_the_record_and_charges_nobody(
        self, fake_dream_db, fake_dream_redis, charges
    ):
        await _in_flight(fake_dream_db, fake_dream_redis)
        entry = _entry("recombine")
        entry.payload["user_id"] = ""

        await handle_dream_batch_result(entry, [])

        assert fake_dream_db.rows["p1"]["status"] is DreamPassStatus.ERRORED
        charges.assert_not_awaited()
        assert state_key("p1") not in fake_dream_redis.hashes
        assert fake_dream_redis.store[_LOCK_KEY] == "tok"


class TestADuplicateOfTheLastPhase:
    async def test_closes_the_row_a_dead_first_delivery_left_applying(
        self, mocker, fake_dream_db, fake_dream_redis, charges
    ):
        """The first delivery claimed apply and died: the duplicate never
        applies, and closes the row instead of leaving it APPLYING."""
        apply = mocker.patch(
            "backend.copilot.dream.apply.apply_operations", AsyncMock()
        )
        await _in_flight(fake_dream_db, fake_dream_redis, landed=2)
        fake_dream_redis.store["dream:applied:p1"] = "1"

        await handle_dream_batch_result(
            _entry("sanitize"), [_row("sanitize", DreamOperations())]
        )

        apply.assert_not_awaited()
        row = fake_dream_db.rows["p1"]
        assert (row["status"], row["error"]) == (
            DreamPassStatus.EXPIRED,
            DUPLICATE_ERROR,
        )
        assert (row["lease_token"], row["lease_expires_at"]) == (None, None)
        assert _charged(charges) == ["consolidate", "recombine", "sanitize"]
        assert _LOCK_KEY not in fake_dream_redis.store

    async def test_leaves_a_row_the_first_delivery_completed(
        self, mocker, fake_dream_db, fake_dream_redis
    ):
        mocker.patch("backend.copilot.dream.apply.apply_operations", AsyncMock())
        await _in_flight(fake_dream_db, fake_dream_redis, landed=2)
        fake_dream_db.rows["p1"]["status"] = DreamPassStatus.COMPLETE
        fake_dream_redis.store["dream:applied:p1"] = "1"

        await handle_dream_batch_result(
            _entry("sanitize"), [_row("sanitize", DreamOperations())]
        )

        row = fake_dream_db.rows["p1"]
        assert (row["status"], row.get("error")) == (DreamPassStatus.COMPLETE, None)


async def _in_flight(fake_dream_db, fake_dream_redis, *, landed: int = 1) -> None:
    """Batch pass p1 holding its lock under ``tok``: its first *landed*
    phases in its state, the next one's batch in flight, its job open."""
    now = datetime.now(timezone.utc)
    await persist_input_bundle(
        "p1",
        DreamInput(
            user_id="u1", group_id=_SCOPE.group_id, window_start=now, window_end=now
        ),
        lock_token="tok",
    )
    fake_dream_redis.store[_LOCK_KEY] = "tok"
    await job_status.write_initial_status(kind="dream_pass", job_id="j1", user_id="u1")
    outputs: tuple[tuple[DreamPhase, BaseModel], ...] = (
        ("consolidate", ConsolidationOutput()),
        ("recombine", RecombinationOutput()),
    )
    for phase, output in outputs[:landed]:
        await write_phase_to_state(pass_id="p1", phase=phase, row=_row(phase, output))
    fake_dream_db.seed(
        DreamPassDraft(
            id="p1",
            user_id="u1",
            scope_key=_SCOPE.scope_key,
            route=DreamPassRoute.ANTHROPIC_BATCH,
            trigger=DreamPassTrigger.CRON,
            status=DreamPassStatus.SUBMITTED,
            phase=DreamPassPhase.RECOMBINE if landed == 1 else DreamPassPhase.SANITIZE,
            lease_token="tok",
            lease_expires_at=now,
        ),
        provider_batch_id="b-rec" if landed == 1 else "b-san",
    )


def _entry(phase: str) -> PendingEntry:
    now = datetime.now(timezone.utc)
    return PendingEntry(
        provider="anthropic",
        provider_batch_id="b-rec" if phase == "recombine" else "b-san",
        callback_namespace="dream_pass",
        submitted_at=now,
        next_poll_at=now,
        payload={
            "user_id": "u1",
            "pass_id": "p1",
            "job_id": "j1",
            "phase": phase,
            "phase_models": _MODELS,
        },
    )


def _row(phase: str, output) -> BatchResultRow:
    return BatchResultRow(
        custom_id=f"p1_{phase}",
        content=output.model_dump_json(),
        input_tokens=10,
        output_tokens=20,
    )


def _charged(charges: AsyncMock) -> list[str]:
    return [call.args[0].job.phase for call in charges.await_args_list]
