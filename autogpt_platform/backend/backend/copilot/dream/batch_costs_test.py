"""Charging a batch pass's landed phases, over the in-memory Redis and store:
each phase is claimed on its own right before it is charged, so a cleanup cut
short resumes with the phases it did not reach and never charges one twice
(Codex's cuts: after the pass is claimed, after its first phase); a charge
the cut lands in finishes on its own; a charge that fails is not tried again;
a pass an earlier build charged whole is not charged again, and one charged
phase by phase keeps an earlier build out. The one gap left, at most once
over exactly once: a process that dies between a phase's claim and its
charge leaves that phase uncharged. The provider and the cost log are
stubbed at their edges."""

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest
from prisma.enums import DreamPassRoute, DreamPassStatus, SubscriptionTier

from backend.copilot.trial_cost_context import TrialCostContext, get_trial_cost_context
from backend.data.pro_activation import UsageActivationState
from backend.util.llm.providers import BatchResultRow

from . import batch_costs as costs_mod
from . import cleanup as cleanup_mod
from . import reaper as reaper_mod
from .batch_costs import charge_landed_phases
from .batch_state import read_state, state_key, write_phase_to_state
from .reaper import reap_expired_passes
from .reaper_cleanup_test import _scale_the_budget
from .reaper_test import _charged, _dead_batch_pass, _seed

_MODELS = {p: "claude-sonnet-5" for p in ("consolidate", "recombine", "sanitize")}
_PASS_KEY = "dream:batch:costs_logged:p1"


@pytest.fixture(autouse=True)
def provider(mocker) -> AsyncMock:
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


@pytest.fixture(autouse=True)
def pass_row(fake_dream_db) -> None:
    """The charged pass's row, live, carrying the usage attribution every
    charge of it goes by (``store.read_cost_context``); a test about a dead
    pass seeds its own over it."""
    _seed(
        fake_dream_db,
        "p1",
        route=DreamPassRoute.ANTHROPIC_BATCH,
        status=DreamPassStatus.SUBMITTED,
        lapsed=-3600,
    )


class TestTheAttributionAChargeGoesBy:
    """Every charge of a pass goes by the usage attribution its row carries,
    whoever makes it and whatever the account's own reads now; a row without
    one is never charged."""

    async def test_is_the_one_the_pass_started_under(
        self, mocker, fake_dream_db, fake_dream_redis, usage_snapshot
    ):
        trial = TrialCostContext(user_id="u1", trial_id="trial-1", generation=None)
        fake_dream_db.rows["p1"]["cost_context"] = trial.model_dump(mode="json")
        usage_snapshot.side_effect = lambda user_id: UsageActivationState(
            user_id=user_id, generation="paid-1", tier=SubscriptionTier.PRO, ready=True
        )
        seen: list[TrialCostContext | None] = []

        async def observe(ctx, usage):
            seen.append(get_trial_cost_context("u1"))

        mocker.patch.object(costs_mod, "record_phase_cost", observe)

        charged = await _charge(await _landed_state())

        assert charged.settled and charged.charged == ["consolidate", "recombine"]
        assert seen == [trial, trial]
        assert get_trial_cost_context("u1") is None
        usage_snapshot.assert_not_awaited()

    @pytest.mark.parametrize(
        "kept",
        [None, {"user_id": "someone-else", "trial_id": None}, {"trial_id": 7}],
        ids=["none", "another_users", "unparseable"],
    )
    async def test_a_row_without_a_usable_one_is_never_charged(
        self, fake_dream_db, fake_dream_redis, charges, kept
    ):
        fake_dream_db.rows["p1"]["cost_context"] = kept

        charged = await _charge(await _landed_state())

        assert charged.settled and charged.charged == []
        charges.assert_not_awaited()
        assert _PASS_KEY not in fake_dream_redis.store

    async def test_a_row_that_cannot_be_read_leaves_the_charge_for_later(
        self, fake_dream_db, fake_dream_redis, charges
    ):
        fake_dream_db.fail = True

        charged = await _charge(await _landed_state())

        assert not charged.settled
        charges.assert_not_awaited()
        assert _PASS_KEY not in fake_dream_redis.store


class TestACleanupCutShortInItsCharge:
    async def test_after_the_pass_is_claimed_the_next_run_charges_every_phase(
        self, mocker, fake_dream_db, fake_dream_redis, charges
    ):
        """The run is cut while the pass's gate, set, never answers; the
        charge goes on until its claim window closes, and gives up there,
        having claimed no phase."""
        _scale_the_budget(mocker, budget=0.3, release=0.05)
        mocker.patch.object(costs_mod, "CLAIM_WINDOW_SECONDS", 0.6)
        await _two_landed(fake_dream_db)
        claim = costs_mod.claim_per_phase_charging

        async def claimed_then_cut(pass_id: str) -> bool:
            await claim(pass_id)
            await asyncio.Event().wait()
            return True

        mocker.patch.object(costs_mod, "claim_per_phase_charging", claimed_then_cut)

        first = await asyncio.wait_for(reap_expired_passes(), 5)

        assert first.outcomes == {"out_of_budget": 1}
        assert costs_mod._CHARGES_IN_FLIGHT, "the charge outlives the run"
        await asyncio.wait_for(asyncio.gather(*costs_mod._CHARGES_IN_FLIGHT), 5)
        assert fake_dream_redis.store[_PASS_KEY] == "per_phase"
        charges.assert_not_awaited()
        assert fake_dream_db.rows["p1"]["cleanup_pending_at"] is not None

        mocker.patch.object(costs_mod, "claim_per_phase_charging", claim)
        second = await reap_expired_passes()

        assert second.outcomes == {"cleaned": 1}
        assert _charged(charges) == ["consolidate", "recombine"]
        assert state_key("p1") not in fake_dream_redis.hashes

    async def test_after_its_first_phase_the_next_run_charges_the_second_only(
        self, mocker, fake_dream_db, fake_dream_redis, charges
    ):
        """The claim window closes while the first phase's cost write runs:
        that phase is charged, the second's claim is given up, and the next
        run charges the second only."""
        _scale_the_budget(mocker, budget=0.3, release=0.05)
        mocker.patch.object(costs_mod, "CLAIM_WINDOW_SECONDS", 0.05)
        await _two_landed(fake_dream_db)
        log = costs_mod._log_phase_cost

        async def slow_consolidate(scope, pass_id, phase, row, models) -> bool:
            if phase == "consolidate":
                await asyncio.sleep(0.1)
            return await log(scope, pass_id, phase, row, models)

        mocker.patch.object(costs_mod, "_log_phase_cost", slow_consolidate)

        first = await asyncio.wait_for(reap_expired_passes(), 5)

        assert first.outcomes == {"retry": 1}
        assert _charged(charges) == ["consolidate"]
        assert "dream:batch:charged:p1:recombine" not in fake_dream_redis.store

        mocker.patch.object(costs_mod, "_log_phase_cost", log)
        second = await reap_expired_passes()

        assert second.outcomes == {"cleaned": 1}
        assert _charged(charges) == ["consolidate", "recombine"]
        assert state_key("p1") not in fake_dream_redis.hashes

    async def test_a_charge_the_cut_lands_in_finishes_on_its_own(
        self, mocker, fake_dream_db, fake_dream_redis, charges
    ):
        """Codex's probe cut the run inside a phase's cost write: the write
        runs in a task of its own, so it finishes after the run has ended,
        and the next run finds the phase charged."""
        _scale_the_budget(mocker, budget=0.3, release=0.05)
        await _two_landed(fake_dream_db)
        log = costs_mod._log_phase_cost

        async def slow_recombine(scope, pass_id, phase, row, models) -> bool:
            if phase == "recombine":
                await asyncio.sleep(0.5)
            return await log(scope, pass_id, phase, row, models)

        mocker.patch.object(costs_mod, "_log_phase_cost", slow_recombine)

        first = await asyncio.wait_for(reap_expired_passes(), 5)

        assert first.outcomes == {"out_of_budget": 1}
        assert _charged(charges) == ["consolidate"]
        await asyncio.wait_for(asyncio.gather(*costs_mod._CHARGES_IN_FLIGHT), 5)
        assert _charged(charges) == ["consolidate", "recombine"]

        second = await reap_expired_passes()

        assert second.outcomes == {"cleaned": 1}
        assert _charged(charges) == ["consolidate", "recombine"]

    async def test_by_its_own_bound_the_claimed_phase_is_charged_on_its_own(
        self, mocker, fake_dream_db, fake_dream_redis, charges
    ):
        """A cost write that stalls keeps the cleanup no longer than its
        charge step's bound: the step is left unfinished (the state kept for
        it), the write lands on its own once it can, and the next run finds
        the phase charged and finishes."""
        mocker.patch.object(cleanup_mod, "CHARGE_STEP_TIMEOUT_SECONDS", 0.05)
        await _dead_batch_pass(fake_dream_db)
        stalled = asyncio.Event()

        async def stalls(*_args, **_kwargs) -> None:
            await stalled.wait()

        charges.side_effect = stalls

        first = await asyncio.wait_for(reap_expired_passes(), 5)

        assert first.outcomes == {"retry": 1}
        assert state_key("p1") in fake_dream_redis.hashes
        stalled.set()
        await asyncio.wait_for(asyncio.gather(*costs_mod._CHARGES_IN_FLIGHT), 5)

        second = await reap_expired_passes()

        assert second.outcomes == {"cleaned": 1}
        assert _charged(charges) == ["consolidate"]
        assert state_key("p1") not in fake_dream_redis.hashes

    async def test_a_crash_between_a_claim_and_its_charge_loses_that_phase(
        self, fake_dream_db, fake_dream_redis, charges
    ):
        """The gap left, written down: a process that died right after
        claiming recombine never charged it, and the claim keeps every later
        cleanup from charging it too."""
        await _two_landed(fake_dream_db)
        fake_dream_redis.store[_PASS_KEY] = "per_phase"
        fake_dream_redis.store["dream:batch:charged:p1:recombine"] = "1"

        run = await reap_expired_passes()

        assert run.outcomes == {"expired": 1}
        assert _charged(charges) == ["consolidate"]
        assert fake_dream_db.rows["p1"]["cleanup_pending_at"] is None

    async def test_a_claim_issued_in_time_is_charged_when_its_reply_comes_late(
        self, mocker, fake_dream_db, fake_dream_redis, charges
    ):
        """Codex's probe: the gate answers late in the claim window, so the
        phase's claim goes out in time; Redis takes it, and its reply comes
        after the window has closed. The window only admits claims, so the
        claim is waited for and the phase charged; the next run finds it
        charged and finishes."""
        mocker.patch.object(costs_mod, "CLAIM_WINDOW_SECONDS", 0.3)
        mocker.patch.object(costs_mod, "CLAIM_TIMEOUT_SECONDS", 0.5)
        mocker.patch.object(cleanup_mod, "CHARGE_STEP_TIMEOUT_SECONDS", 0.1)
        await _dead_batch_pass(fake_dream_db)
        gate = costs_mod.claim_per_phase_charging

        async def slow_gate(pass_id: str) -> bool:
            await asyncio.sleep(0.2)
            return await gate(pass_id)

        claim = "dream:batch:charged:p1:consolidate"
        set_ = fake_dream_redis.set
        late: list[str] = []

        async def late_reply(key: str, *args, **kwargs):
            taken = await set_(key, *args, **kwargs)
            if key == claim and taken:
                await asyncio.sleep(0.15)
                late.append(key)
            return taken

        mocker.patch.object(costs_mod, "claim_per_phase_charging", slow_gate)
        mocker.patch.object(fake_dream_redis, "set", late_reply)

        first = await asyncio.wait_for(reap_expired_passes(), 5)
        await asyncio.wait_for(asyncio.gather(*costs_mod._CHARGES_IN_FLIGHT), 5)

        assert first.outcomes == {"retry": 1}
        assert late == [claim]
        assert _charged(charges) == ["consolidate"]

        mocker.patch.object(costs_mod, "claim_per_phase_charging", gate)
        second = await reap_expired_passes()

        assert second.outcomes == {"cleaned": 1}
        assert _charged(charges) == ["consolidate"]
        assert fake_dream_db.rows["p1"]["cleanup_pending_at"] is None


class TestAPhaseCharge:
    @pytest.mark.parametrize(
        "error", ["max_tokens", "invalid output shape", "provider error"]
    )
    async def test_failed_model_phases_are_absorbed_when_the_reaper_settles(
        self, fake_dream_db, fake_dream_redis, charges, error
    ):
        await _dead_batch_pass(fake_dream_db)
        await write_phase_to_state(
            pass_id="p1",
            phase="recombine",
            row=BatchResultRow(
                custom_id="p1_recombine",
                content="{}",
                input_tokens=30,
                output_tokens=40,
                error=error,
            ),
            model=_MODELS["recombine"],
        )

        await reap_expired_passes()
        await reap_expired_passes()

        assert _charged(charges) == ["consolidate"]
        assert fake_dream_db.rows["p1"]["cleanup_pending_at"] is None
        assert state_key("p1") not in fake_dream_redis.hashes

    async def test_is_made_once_whoever_asks_and_however_often(
        self, fake_dream_redis, charges
    ):
        state = await _landed_state()

        first = await _charge(state)
        again = await _charge(state)

        assert (first.charged, first.settled) == (["consolidate", "recombine"], True)
        assert (again.charged, again.settled) == ([], True)
        assert _charged(charges) == ["consolidate", "recombine"]

    async def test_that_fails_is_not_tried_again(
        self, fake_dream_redis, charges, caplog
    ):
        """The trial ledger or the weekly counter may have moved before the
        cost-log write raised; a second attempt could charge them twice."""
        state = await _landed_state()
        charges.side_effect = [RuntimeError("cost log down"), None]

        with caplog.at_level(logging.ERROR):
            first = await _charge(state)
        again = await _charge(state)

        assert (first.charged, first.settled) == (["recombine"], True)
        assert (again.charged, again.settled) == ([], True)
        assert charges.await_count == 2
        assert "Failed to log batch cost for pass=p1 phase=consolidate" in caplog.text

    async def test_whose_claim_cannot_be_made_is_left_for_the_next_cleanup(
        self, monkeypatch, fake_dream_redis, charges
    ):
        state = await _landed_state()
        claim = costs_mod.claim_phase_charge

        async def down_for_recombine(pass_id: str, phase: str) -> bool:
            if phase == "recombine":
                raise ConnectionError("redis down")
            return await claim(pass_id, phase)

        monkeypatch.setattr(costs_mod, "claim_phase_charge", down_for_recombine)
        first = await _charge(state)
        monkeypatch.setattr(costs_mod, "claim_phase_charge", claim)
        again = await _charge(state)

        assert (first.charged, first.settled) == (["consolidate"], False)
        assert (again.charged, again.settled) == (["recombine"], True)
        assert _charged(charges) == ["consolidate", "recombine"]


class TestTwoBuildsSideBySide:
    async def test_a_pass_an_earlier_build_charged_whole_is_not_charged_again(
        self, fake_dream_redis, charges
    ):
        fake_dream_redis.store[_PASS_KEY] = "1"

        charged = await _charge(await _landed_state())

        assert (charged.charged, charged.settled) == ([], True)
        charges.assert_not_awaited()

    async def test_an_earlier_build_finds_a_pass_charged_phase_by_phase_taken(
        self, fake_dream_redis, charges
    ):
        """An earlier build claims the pass-level key with SETNX before it
        charges every phase: this build's claim of it, under its own value,
        makes that SETNX fail, so the earlier build charges nothing."""
        await _charge(await _landed_state())

        earlier_build = await fake_dream_redis.set(_PASS_KEY, "1", nx=True)

        assert earlier_build is None
        assert fake_dream_redis.store[_PASS_KEY] == "per_phase"
        assert charges.await_count == 2


async def _two_landed(fake_dream_db) -> None:
    """A dead batch pass with consolidate and recombine in its state."""
    await _dead_batch_pass(fake_dream_db)
    await write_phase_to_state(
        pass_id="p1",
        phase="recombine",
        row=BatchResultRow(
            custom_id="p1_recombine", content="{}", input_tokens=30, output_tokens=40
        ),
    )


async def _landed_state() -> dict:
    """The state of a pass with consolidate and recombine landed."""
    for phase in ("consolidate", "recombine"):
        await write_phase_to_state(
            pass_id="p1",
            phase=phase,
            row=BatchResultRow(
                custom_id=f"p1_{phase}", content="{}", input_tokens=10, output_tokens=20
            ),
        )
    return await read_state("p1")


async def _charge(state: dict) -> costs_mod.PhaseCharges:
    return await charge_landed_phases(
        user_id="u1", expert_id=None, pass_id="p1", state=state, phase_models=_MODELS
    )
