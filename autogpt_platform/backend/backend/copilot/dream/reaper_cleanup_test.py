"""The reaper's cleanup after the passes it closes, and its budget, over the
in-memory store and Redis: a cleanup cut short at any step, or by the budget,
is resumed by the next run and finishes once (one charge, the state and
bundle gone, the mark and the kept token cleared); a run never outlasts its
budget, the release of its hold on a scope included, and that release
finishes when the budget cuts into it; the budget runs from before the
listing; every row gets its own line. The provider and the charges are
stubbed at their edges; the lock, the gates and the row transitions are the
real ones."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from prisma.enums import DreamPassRoute, DreamPassStatus

from . import reaper as reaper_mod
from .batch_state import state_key
from .batch_submit import input_bundle_key
from .reaper import reap_expired_passes
from .reaper_test import _LOCK_KEY, _charged, _dead_batch_pass, _seed

# Each cleanup step, by the name the reaper calls it under.
_STEPS = (
    "cancel_provider_batch",
    "read_state_or_none",
    "release_dream_lock",
    "best_effort_cleanup",
    "record_cleanup_finished",
)


@pytest.fixture(autouse=True)
def provider_cancel(mocker) -> AsyncMock:
    mocker.patch(
        "backend.copilot.dream.provider_batch.anthropic_api_key", return_value="k"
    )
    mocker.patch.object(
        reaper_mod,
        "phase_models_for_config",
        return_value={p: "claude-sonnet-5" for p in ("consolidate", "recombine")},
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


class TestACleanupCutShort:
    @pytest.mark.parametrize("step", _STEPS)
    async def test_at_any_step_is_finished_by_the_next_run_once(
        self, mocker, fake_dream_db, fake_dream_redis, charges, step
    ):
        """The dead pass still holds its lock, so every step has work to do;
        the run is cut at *step*, after the row closed. The next run lists
        the row by its mark and does every step again, the done ones doing
        nothing: one charge in all, the lock and the state gone, the mark and
        the kept token cleared."""
        await _dead_batch_pass(fake_dream_db)
        fake_dream_redis.store[_LOCK_KEY] = "dead-token"
        _fail_once(mocker, step)

        first = await reap_expired_passes()

        assert first.outcomes == {"failed": 1}
        row = fake_dream_db.rows["p1"]
        assert row["status"] is DreamPassStatus.EXPIRED
        assert row["cleanup_pending_at"] is not None
        assert row["lease_token"] == "dead-token"

        second = await reap_expired_passes()

        assert (second.listed, second.outcomes) == (1, {"cleaned": 1})
        assert _charged(charges) == ["consolidate"]
        assert state_key("p1") not in fake_dream_redis.hashes
        assert input_bundle_key("p1") not in fake_dream_redis.store
        assert _LOCK_KEY not in fake_dream_redis.store
        row = fake_dream_db.rows["p1"]
        assert (row["cleanup_pending_at"], row["lease_token"]) == (None, None)
        assert row["status"] is DreamPassStatus.EXPIRED
        assert (await reap_expired_passes()).listed == 0

    async def test_by_the_budget_after_the_close_is_resumed_next_run(
        self, mocker, fake_dream_db, fake_dream_redis, provider_cancel, charges
    ):
        """Codex's case: the budget runs out while the provider cancel of a
        row the run just closed hangs. The row keeps its mark, so the next
        run finds it again and charges its landed phase, once."""
        _scale_the_budget(mocker, budget=0.3, release=0.05)
        await _dead_batch_pass(fake_dream_db)

        async def never_answers(**_kwargs) -> bool:
            await asyncio.Event().wait()
            return True

        provider_cancel.side_effect = never_answers

        first = await asyncio.wait_for(reap_expired_passes(), 5)

        assert first.outcomes == {"out_of_budget": 1}
        assert fake_dream_db.rows["p1"]["status"] is DreamPassStatus.EXPIRED
        assert fake_dream_db.rows["p1"]["cleanup_pending_at"] is not None
        assert state_key("p1") in fake_dream_redis.hashes
        charges.assert_not_awaited()

        provider_cancel.side_effect = None
        second = await reap_expired_passes()

        assert second.outcomes == {"cleaned": 1}
        assert _charged(charges) == ["consolidate"]
        assert state_key("p1") not in fake_dream_redis.hashes
        assert fake_dream_db.rows["p1"]["cleanup_pending_at"] is None


class TestTheListing:
    async def test_takes_the_pending_cleanups_first_within_the_cap(
        self, mocker, fake_dream_db
    ):
        mocker.patch.object(reaper_mod, "REAPER_ROW_LIMIT", 2)
        for pass_id, lapsed_hours in (("new", 1), ("old", 5)):
            _seed(
                fake_dream_db,
                pass_id,
                route=DreamPassRoute.SYNC,
                lapsed=lapsed_hours * 3600,
            )
        _seed(
            fake_dream_db,
            "closed",
            route=DreamPassRoute.SYNC,
            status=DreamPassStatus.EXPIRED,
            cleanup_pending_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )

        run = await reap_expired_passes()

        assert (run.listed, run.outcomes) == (2, {"cleaned": 1, "expired": 1})
        statuses = {pid: row["status"] for pid, row in fake_dream_db.rows.items()}
        assert statuses["old"] is DreamPassStatus.EXPIRED
        assert statuses["new"] is DreamPassStatus.RUNNING
        assert fake_dream_db.rows["closed"]["cleanup_pending_at"] is None


class TestTheBudget:
    async def test_bounds_the_release_of_the_reapers_hold_too(
        self, mocker, fake_dream_db, fake_dream_redis, provider_cancel, caplog
    ):
        """The budget cuts a row whose provider cancel hangs, and the release
        of the scope the reaper held hangs too: the run still ends at its
        budget, and the hold is left to lapse on its own TTL."""
        _scale_the_budget(mocker, budget=0.4, release=0.1)
        await _dead_batch_pass(fake_dream_db)

        async def never_answers(**_kwargs) -> bool:
            await asyncio.Event().wait()
            return True

        provider_cancel.side_effect = never_answers
        _hang_the_release_of_the_hold(mocker)
        loop = asyncio.get_running_loop()

        started = loop.time()
        with caplog.at_level(logging.WARNING):
            reaping = asyncio.create_task(reap_expired_passes())
            await asyncio.wait({reaping}, timeout=2)

        assert reaping.done(), "the run outlasted its budget"
        assert loop.time() - started < 0.4 + 0.15
        assert reaping.result().outcomes == {"out_of_budget": 1}
        assert fake_dream_redis.store[_LOCK_KEY].startswith("reaper:")
        assert "could not give back scope" in caplog.text

    async def test_lets_a_release_it_cuts_into_finish(
        self, mocker, fake_dream_db, fake_dream_redis
    ):
        """The row finishes, and the budget runs out while the reaper gives
        its hold back: the release is finished all the same, inside its own
        bound, so the scope is not left held."""
        _scale_the_budget(mocker, budget=0.35, release=0.3)
        _seed(fake_dream_db, "s1", route=DreamPassRoute.SYNC)
        release = reaper_mod.release_dream_lock

        async def slow_for_the_hold(scope, token: str | None) -> None:
            if token is not None and token.startswith("reaper:"):
                await asyncio.sleep(0.15)
            await release(scope, token)

        mocker.patch.object(reaper_mod, "release_dream_lock", slow_for_the_hold)

        run = await asyncio.wait_for(reap_expired_passes(), 5)

        assert run.outcomes == {"expired": 1}
        assert _LOCK_KEY not in fake_dream_redis.store

    async def test_runs_from_before_the_listing(self, mocker, fake_dream_db):
        """A slow listing spends the budget too: with less than a row's
        reserve left once the rows are in, none is started."""
        _scale_the_budget(mocker, budget=1.0, release=0.1, reserve=0.5)
        _seed(fake_dream_db, "s1", route=DreamPassRoute.SYNC)
        list_pending = reaper_mod.read_pending_cleanups

        async def slow_listing(*, limit: int):
            await asyncio.sleep(0.5)
            return await list_pending(limit=limit)

        mocker.patch.object(reaper_mod, "read_pending_cleanups", slow_listing)

        run = await reap_expired_passes()

        assert run.outcomes == {"out_of_budget": 1}
        assert fake_dream_db.rows["s1"]["status"] is DreamPassStatus.RUNNING


async def test_every_row_gets_its_own_line(mocker, fake_dream_db, caplog):
    """Expired and moved at INFO; failed, and each row the budget did not
    reach, at WARNING; then the counts."""
    for pass_id in ("s1", "s2", "s3"):
        _seed(fake_dream_db, pass_id, route=DreamPassRoute.SYNC)
    write_stop = reaper_mod.write_stop

    async def down_for_s1_and_then_out_of_time(pass_id, update):
        if pass_id == "s1":
            raise TimeoutError("dream pass database stalled")
        mocker.patch.object(reaper_mod, "ROW_RESERVE_SECONDS", 10_000)
        return await write_stop(pass_id, update)

    mocker.patch.object(reaper_mod, "write_stop", down_for_s1_and_then_out_of_time)

    with caplog.at_level(logging.INFO):
        run = await reap_expired_passes()

    assert run.outcomes == {"expired": 1, "failed": 1, "out_of_budget": 1}
    lines = {(r.levelname, r.getMessage().split(";")[0]) for r in caplog.records}
    assert ("WARNING", "Dream pass reaper: pass s1 failed") in lines
    assert any(
        level == "INFO" and message.startswith("Dream pass reaper: pass s2 expired")
        for level, message in lines
    )
    assert (
        "WARNING",
        "Dream pass reaper: pass s3 not finished within the budget",
    ) in lines
    assert "Dream pass reaper: 3 listed; expired=1, failed=1, out_of_budget=1" in (
        caplog.text
    )


def _fail_once(mocker, step: str) -> None:
    """Make the reaper's *step* raise the first time it is called, as a run
    cut short there would stop, and work as before after that."""
    real = getattr(reaper_mod, step)
    calls = 0

    async def once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError(f"cut short at {step}")
        return await real(*args, **kwargs)

    mocker.patch.object(reaper_mod, step, once)


def _scale_the_budget(
    mocker, *, budget: float, release: float, reserve: float = 0.01
) -> None:
    mocker.patch.object(reaper_mod, "REAPER_BUDGET_SECONDS", budget)
    mocker.patch.object(reaper_mod, "RELEASE_TIMEOUT_SECONDS", release)
    mocker.patch.object(reaper_mod, "ROW_RESERVE_SECONDS", reserve)
    mocker.patch(
        "backend.copilot.dream.provider_batch.PROVIDER_CANCEL_TIMEOUT_SECONDS", 30
    )


def _hang_the_release_of_the_hold(mocker) -> None:
    """Only the compare-and-delete of the reaper's own hold never answers."""
    release = reaper_mod.release_dream_lock

    async def hangs_for_the_hold(scope, token: str | None) -> None:
        if token is not None and token.startswith("reaper:"):
            await asyncio.Event().wait()
        await release(scope, token)

    mocker.patch.object(reaper_mod, "release_dream_lock", hangs_for_the_hold)
