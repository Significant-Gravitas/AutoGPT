"""The dream pass reaper over the in-memory store and Redis: every branch of
closing a pass that outlived its lease. The provider and the charges are
stubbed at their edges; the lock, the gates, the batch state and the row
transitions are the real ones."""

import asyncio
import logging
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
from backend.data.dream_pass_models import DreamPassDraft
from backend.util.llm.providers import BatchResultRow

from . import reaper as reaper_mod
from .batch_state import STATE_TTL_SECONDS, state_key, write_phase_to_state
from .batch_submit import INPUT_TTL_SECONDS, input_bundle_key, persist_input_bundle
from .fetch import DreamInput
from .locks import BATCH_LOCK_TTL_SECONDS, DEFAULT_LOCK_TTL_SECONDS
from .reaper import REAP_GRACE_SECONDS, REAPER_INTERVAL_MINUTES, reap_expired_passes
from .schemas import ConsolidationOutput

_SCOPE = MemoryScope.for_user("u1")
_LOCK_KEY = _SCOPE.redis_key("dream_lock")
_MODELS = {p: "claude-sonnet-5" for p in ("consolidate", "recombine", "sanitize")}


@pytest.fixture(autouse=True)
def provider_cancel(mocker) -> AsyncMock:
    mocker.patch(
        "backend.copilot.dream.provider_batch.anthropic_api_key", return_value="k"
    )
    mocker.patch.object(reaper_mod, "phase_models_for_config", return_value=_MODELS)
    return mocker.patch(
        "backend.copilot.dream.provider_batch.cancel_batch",
        AsyncMock(return_value=True),
    )


@pytest.fixture(autouse=True)
def charges(mocker) -> AsyncMock:
    return mocker.patch(
        "backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock()
    )


class TestAPassThatOutlivedItsLease:
    async def test_is_expired_charged_and_cleaned_up(
        self, fake_dream_db, fake_dream_redis, provider_cancel, charges, caplog
    ):
        await _dead_batch_pass(fake_dream_db)
        holders: list[str | None] = []

        async def cancel(**_kwargs) -> bool:
            holders.append(fake_dream_redis.store.get(_LOCK_KEY))
            return True

        provider_cancel.side_effect = cancel

        with caplog.at_level(logging.INFO):
            run = await reap_expired_passes()

        assert (run.listed, run.outcomes) == (1, {"expired": 1})
        # The reaper held the free scope while it cleaned up, then let go.
        assert len(holders) == 1 and (holders[0] or "").startswith("reaper:")
        row = fake_dream_db.rows["p1"]
        assert (row["status"], row["cancel_generation"]) == (
            DreamPassStatus.EXPIRED,
            1,
        )
        assert row["error"].startswith("recombine: the lease lapsed at ")
        assert row["error"].endswith("; closed by the reaper")
        assert (row["lease_token"], row["lease_expires_at"]) == (None, None)
        assert row["input_bundle"] is None
        provider_cancel.assert_awaited_once()
        assert provider_cancel.await_args.kwargs["provider_batch_id"] == "b-rec"
        assert _charged(charges) == ["consolidate"]
        assert state_key("p1") not in fake_dream_redis.hashes
        assert input_bundle_key("p1") not in fake_dream_redis.store
        assert _LOCK_KEY not in fake_dream_redis.store
        assert "Dream pass reaper: pass p1 expired (recombine:" in caplog.text
        assert "Dream pass reaper: 1 listed; expired=1" in caplog.text

    async def test_within_its_grace_is_left_open(self, fake_dream_db, provider_cancel):
        """One sync lock TTL past its lease a pass is taken for dead, not
        before: a pass slow to renew is left alone."""
        assert REAP_GRACE_SECONDS == DEFAULT_LOCK_TTL_SECONDS
        await _dead_batch_pass(fake_dream_db, lapsed=25 * 60)

        run = await reap_expired_passes()

        assert (run.listed, run.outcomes) == (0, {})
        assert fake_dream_db.rows["p1"]["status"] is DreamPassStatus.SUBMITTED
        provider_cancel.assert_not_awaited()

    async def test_whose_scope_another_pass_holds_is_closed_and_that_lock_kept(
        self, fake_dream_db, fake_dream_redis
    ):
        await _dead_batch_pass(fake_dream_db)
        fake_dream_redis.store[_LOCK_KEY] = "newer-token"

        run = await reap_expired_passes()

        assert run.outcomes == {"expired": 1}
        assert fake_dream_db.rows["p1"]["status"] is DreamPassStatus.EXPIRED
        assert fake_dream_redis.store[_LOCK_KEY] == "newer-token"

    async def test_whose_own_lock_outlived_its_lease_has_it_released(
        self, fake_dream_db, fake_dream_redis
    ):
        """A renewal extended the lock but its row write was lost: the lease
        on the row is the authority, and the lock under its token goes."""
        await _dead_batch_pass(fake_dream_db)
        fake_dream_redis.store[_LOCK_KEY] = "dead-token"

        run = await reap_expired_passes()

        assert run.outcomes == {"expired": 1}
        assert _LOCK_KEY not in fake_dream_redis.store

    async def test_written_since_it_was_read_is_left_alone(
        self, mocker, fake_dream_db, fake_dream_redis, provider_cancel, charges
    ):
        """The pass renewed (or closed) between the read and the expiry: the
        compare-and-set refuses, and nothing of the pass is touched."""
        await _dead_batch_pass(fake_dream_db)
        listed = await reaper_mod.read_expired_passes(
            datetime.now(timezone.utc), limit=10
        )
        fake_dream_db.rows["p1"]["updated_at"] = datetime.now(timezone.utc)
        mocker.patch.object(
            reaper_mod, "read_expired_passes", AsyncMock(return_value=listed)
        )

        run = await reap_expired_passes()

        assert run.outcomes == {"moved": 1}
        assert fake_dream_db.rows["p1"]["status"] is DreamPassStatus.SUBMITTED
        provider_cancel.assert_not_awaited()
        charges.assert_not_awaited()
        assert state_key("p1") in fake_dream_redis.hashes
        assert _LOCK_KEY not in fake_dream_redis.store

    async def test_whose_provider_cancel_fails_is_closed_all_the_same(
        self, fake_dream_db, fake_dream_redis, provider_cancel, charges
    ):
        await _dead_batch_pass(fake_dream_db)
        provider_cancel.side_effect = ConnectionError("anthropic unreachable")

        run = await reap_expired_passes()

        assert run.outcomes == {"expired": 1}
        assert fake_dream_db.rows["p1"]["status"] is DreamPassStatus.EXPIRED
        assert _charged(charges) == ["consolidate"]
        assert state_key("p1") not in fake_dream_redis.hashes

    async def test_already_charged_is_not_charged_again(
        self, fake_dream_db, fake_dream_redis, charges, caplog
    ):
        await _dead_batch_pass(fake_dream_db)
        fake_dream_redis.store["dream:batch:costs_logged:p1"] = "1"

        with caplog.at_level(logging.INFO):
            run = await reap_expired_passes()

        assert run.outcomes == {"expired": 1}
        charges.assert_not_awaited()
        assert "charged nothing (already charged)" in caplog.text

    async def test_that_died_applying_is_closed_without_applying_again(
        self, mocker, fake_dream_db, fake_dream_redis, charges
    ):
        """It claimed apply, then its process died: the reaper closes the
        row, naming apply, and never calls apply itself."""
        apply = mocker.patch(
            "backend.copilot.dream.apply.apply_operations", AsyncMock()
        )
        await _dead_batch_pass(
            fake_dream_db,
            status=DreamPassStatus.APPLYING,
            phase=DreamPassPhase.APPLY,
        )
        fake_dream_redis.store["dream:applied:p1"] = "1"

        run = await reap_expired_passes()

        assert run.outcomes == {"expired": 1}
        row = fake_dream_db.rows["p1"]
        assert row["status"] is DreamPassStatus.EXPIRED
        assert row["error"].startswith("apply: the lease lapsed at ")
        assert "while applying; closed by the reaper without applying again" in (
            row["error"]
        )
        apply.assert_not_awaited()
        assert _charged(charges) == ["consolidate"]

    async def test_of_the_sync_route_is_closed_with_nothing_to_cancel(
        self, fake_dream_db, provider_cancel, charges
    ):
        _seed(fake_dream_db, "s1", route=DreamPassRoute.SYNC)

        run = await reap_expired_passes()

        assert run.outcomes == {"expired": 1}
        assert fake_dream_db.rows["s1"]["status"] is DreamPassStatus.EXPIRED
        provider_cancel.assert_not_awaited()
        charges.assert_not_awaited()


def test_a_batch_passs_state_outlives_its_lease_until_the_reaper_comes():
    """A dead batch pass's state was written at its last callback, with its
    lease: the reaper takes it for dead a grace after the lease, and may come
    one interval later still. Its landed phases are charged and its keys
    deleted only if they are still there."""
    reaped_by = (
        BATCH_LOCK_TTL_SECONDS + REAP_GRACE_SECONDS + REAPER_INTERVAL_MINUTES * 60
    )
    assert min(STATE_TTL_SECONDS, INPUT_TTL_SECONDS) > reaped_by


class TestARun:
    async def test_counts_each_outcome_and_goes_on_past_a_row_that_fails(
        self, mocker, fake_dream_db, caplog
    ):
        for pass_id in ("s1", "s2"):
            _seed(fake_dream_db, pass_id, route=DreamPassRoute.SYNC)
        write_stop = reaper_mod.write_stop

        async def down_for_s1(pass_id, update):
            if pass_id == "s1":
                raise TimeoutError("dream pass database stalled")
            return await write_stop(pass_id, update)

        mocker.patch.object(reaper_mod, "write_stop", down_for_s1)

        with caplog.at_level(logging.INFO):
            run = await reap_expired_passes()

        assert run.outcomes == {"expired": 1, "failed": 1}
        assert fake_dream_db.rows["s1"]["status"] is DreamPassStatus.RUNNING
        assert fake_dream_db.rows["s2"]["status"] is DreamPassStatus.EXPIRED
        assert "Dream pass reaper: 2 listed; expired=1, failed=1" in caplog.text

    async def test_never_outlasts_its_budget(
        self, mocker, fake_dream_db, fake_dream_redis, provider_cancel
    ):
        """A provider cancel that never answers holds the first row: the
        budget cancels it, the rows not reached are left for the next run,
        and the scope the reaper held is let go."""
        mocker.patch.object(reaper_mod, "REAPER_BUDGET_SECONDS", 0.5)
        mocker.patch.object(reaper_mod, "ROW_RESERVE_SECONDS", 0.1)
        mocker.patch(
            "backend.copilot.dream.provider_batch.PROVIDER_CANCEL_TIMEOUT_SECONDS",
            30,
        )
        for pass_id in ("p1", "p2", "p3"):
            await _dead_batch_pass(fake_dream_db, pass_id=pass_id)

        async def never_answers(**_kwargs) -> bool:
            await asyncio.Event().wait()
            return True

        provider_cancel.side_effect = never_answers
        loop = asyncio.get_running_loop()

        started = loop.time()
        run = await asyncio.wait_for(reap_expired_passes(), 10)

        assert loop.time() - started < 5
        assert (run.listed, run.outcomes) == (3, {"out_of_budget": 3})
        assert fake_dream_db.rows["p2"]["status"] is DreamPassStatus.SUBMITTED
        assert _LOCK_KEY not in fake_dream_redis.store

    async def test_takes_the_oldest_lapses_first_up_to_its_limit(
        self, mocker, fake_dream_db
    ):
        mocker.patch.object(reaper_mod, "REAPER_ROW_LIMIT", 2)
        for pass_id, lapsed_hours in (("new", 1), ("old", 5), ("older", 9)):
            _seed(
                fake_dream_db,
                pass_id,
                route=DreamPassRoute.SYNC,
                lapsed=lapsed_hours * 3600,
            )

        run = await reap_expired_passes()

        assert run.outcomes == {"expired": 2}
        statuses = {pid: row["status"] for pid, row in fake_dream_db.rows.items()}
        assert statuses == {
            "new": DreamPassStatus.RUNNING,
            "old": DreamPassStatus.EXPIRED,
            "older": DreamPassStatus.EXPIRED,
        }

    async def test_a_store_that_cannot_list_ends_the_run_quietly(
        self, fake_dream_db, caplog
    ):
        fake_dream_db.fail = True

        with caplog.at_level(logging.WARNING):
            run = await reap_expired_passes()

        assert (run.listed, run.outcomes) == (0, {})
        assert "Dream pass reaper: the run stopped short" in caplog.text


async def _dead_batch_pass(
    fake_dream_db,
    *,
    pass_id: str = "p1",
    status: DreamPassStatus = DreamPassStatus.SUBMITTED,
    phase: DreamPassPhase = DreamPassPhase.RECOMBINE,
    lapsed: int = REAP_GRACE_SECONDS + 600,
) -> None:
    """A batch pass that died at *phase*: consolidate landed in its state,
    recombine's batch in flight, its bundle kept, its lease lapsed *lapsed*
    seconds ago."""
    now = datetime.now(timezone.utc)
    await persist_input_bundle(
        pass_id,
        DreamInput(
            user_id="u1", group_id=_SCOPE.group_id, window_start=now, window_end=now
        ),
        lock_token="dead-token",
    )
    await write_phase_to_state(
        pass_id=pass_id,
        phase="consolidate",
        row=BatchResultRow(
            custom_id=f"{pass_id}_consolidate",
            content=ConsolidationOutput().model_dump_json(),
            input_tokens=10,
            output_tokens=20,
        ),
    )
    _seed(
        fake_dream_db,
        pass_id,
        route=DreamPassRoute.ANTHROPIC_BATCH,
        status=status,
        phase=phase,
        lapsed=lapsed,
        provider_batch_id="b-rec",
    )


def _seed(
    fake_dream_db,
    pass_id: str,
    *,
    route: DreamPassRoute,
    status: DreamPassStatus = DreamPassStatus.RUNNING,
    phase: DreamPassPhase = DreamPassPhase.RECOMBINE,
    lapsed: int = REAP_GRACE_SECONDS + 600,
    provider_batch_id: str | None = None,
) -> None:
    """An open row whose lease lapsed *lapsed* seconds ago and that nothing
    has written since."""
    lease_end = datetime.now(timezone.utc) - timedelta(seconds=lapsed)
    fake_dream_db.seed(
        DreamPassDraft(
            id=pass_id,
            user_id="u1",
            scope_key=_SCOPE.scope_key,
            route=route,
            trigger=DreamPassTrigger.CRON,
            status=status,
            phase=phase,
            lease_token="dead-token",
            lease_expires_at=lease_end,
        ),
        updated_at=lease_end - timedelta(minutes=1),
        provider_batch_id=provider_batch_id,
    )


def _charged(charges: AsyncMock) -> list[str]:
    return [call.args[0].job.phase for call in charges.await_args_list]
