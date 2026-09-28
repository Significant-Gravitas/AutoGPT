"""The lock checks right before a dream pass applies, on both routes, with the
real dream lock, callbacks and orchestrator over the in-memory Redis and
store: a pass whose lock lapsed, and was maybe taken by a newer pass, never
applies, and neither does one that cannot prove it still holds its lock, so at
most one pass of a scope applies. Only apply and the models are stubbed."""

import asyncio
from datetime import datetime, timedelta, timezone
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
from backend.copilot.inference.complete import StructuredCompletion
from backend.copilot.inference.context import InferenceUsage, RouteDecision
from backend.data.dream_pass_models import DreamPassDraft
from backend.executor.batch_executor import PendingEntry
from backend.util.llm.providers import BatchResultRow

from . import batch_callbacks as batch_callbacks_mod
from . import batch_outcome as batch_outcome_mod
from . import job_status
from . import lease as lease_mod
from . import locks as locks_mod
from . import orchestrator as orchestrator_mod
from .batch_callbacks import handle_dream_batch_result
from .batch_state import state_key, write_phase_to_state
from .batch_submit import input_bundle_key, persist_input_bundle
from .cancel import LOCK_LOST_ERROR, cancel_dream_pass
from .fetch import DreamInput, EpisodeRow
from .schemas import (
    ConsolidatedFact,
    ConsolidationOutput,
    DreamOperations,
    DreamPhase,
    RecombinationOutput,
)

_SCOPE = MemoryScope.for_user("u1")
_LOCK_KEY = _SCOPE.redis_key("dream_lock")
_PHASE_MODELS = {
    "consolidate": "claude-sonnet-5",
    "recombine": "claude-opus-5-5",
    "sanitize": "claude-sonnet-5",
}
_OPS = DreamOperations(
    writes=[ConsolidatedFact(content="Nick ships on Fridays", confidence=0.9)],
    summary_for_user="ok",
)
_BILLED = InferenceUsage(
    model="m",
    input_tokens=100,
    output_tokens=20,
    cost_usd=0.001,
    cost_source="provider",
    payer="platform_allowance",
)


@pytest.fixture
def apply(mocker) -> AsyncMock:
    """Apply on both routes: one mock, so the calls show which pass applied."""
    applied = AsyncMock(return_value={"session_id": "s", "consolidated_count": 1})
    mocker.patch.object(orchestrator_mod, "apply_operations", applied)
    mocker.patch("backend.copilot.dream.apply.apply_operations", applied)
    return applied


@pytest.fixture
def charges(mocker) -> AsyncMock:
    return mocker.patch(
        "backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock()
    )


@pytest.fixture
def sync_pass(mocker) -> None:
    """A sync pass for the same scope, its three phases answering in turn."""
    mocker.patch.object(orchestrator_mod, "resolve_route", side_effect=_route)
    mocker.patch.object(
        orchestrator_mod, "resolve_dream_execution_path", return_value="sync_baseline"
    )
    mocker.patch.object(
        orchestrator_mod, "is_feature_enabled", AsyncMock(return_value=False)
    )
    mocker.patch.object(
        orchestrator_mod, "check_dream_budget", AsyncMock(return_value=(True, None))
    )
    mocker.patch.object(
        orchestrator_mod,
        "record_phase_cost",
        AsyncMock(side_effect=lambda ctx, usage: usage),
    )
    mocker.patch.object(
        orchestrator_mod, "gather_dream_input", AsyncMock(return_value=_input())
    )
    answers = (ConsolidationOutput(), RecombinationOutput(), _OPS)
    mocker.patch.object(
        orchestrator_mod,
        "structured_complete",
        AsyncMock(
            side_effect=[StructuredCompletion(value=a, usage=_BILLED) for a in answers]
        ),
    )


class TestTheBatchRoute:
    async def test_a_callback_paused_at_its_gate_never_applies_after_its_replacement(
        self, fake_dream_db, fake_dream_redis, apply, charges, sync_pass, mocker
    ):
        """Codex's interleaving: the last callback has checked its row and
        waits on the apply gate; its lock lapses; a forced admin pass takes
        the scope, expires the old row and applies. The old callback then
        claims its gate, finds the lock gone and does not apply."""
        await _seed_batch_pass(fake_dream_db, fake_dream_redis)
        at_gate, resume = asyncio.Event(), asyncio.Event()
        claim = batch_callbacks_mod.claim_apply_gate

        async def paused_claim(pass_id: str):
            at_gate.set()
            await resume.wait()
            return await claim(pass_id)

        mocker.patch.object(batch_callbacks_mod, "claim_apply_gate", paused_claim)
        old = asyncio.create_task(_deliver_sanitize())
        await asyncio.wait_for(at_gate.wait(), 5)
        await fake_dream_redis.delete(_LOCK_KEY)

        newer = await orchestrator_mod.execute_dream_pass(
            "u1", trigger="admin", force=True
        )
        resume.set()
        await asyncio.wait_for(old, 5)

        assert newer.error is None and not newer.skipped
        assert [call.args[1] for call in apply.await_args_list] == [newer.pass_id]
        error = f"expired: forced by an admin-triggered dream pass {newer.pass_id}"
        await _assert_ended_without_applying(fake_dream_db, fake_dream_redis, error)
        assert fake_dream_db.rows["p1"]["status"] is DreamPassStatus.EXPIRED
        assert fake_dream_db.rows[newer.pass_id]["status"] is DreamPassStatus.COMPLETE
        assert _charged(charges) == ["consolidate", "recombine", "sanitize"]

    async def test_a_callback_whose_lock_a_newer_pass_holds_leaves_it_alone(
        self, fake_dream_db, fake_dream_redis, apply, charges, mocker
    ):
        """The lock was already a newer pass's when the last phase landed:
        the callback's lease renewal finds it and ends the pass before it
        even claims apply."""
        await _seed_batch_pass(fake_dream_db, fake_dream_redis)
        fake_dream_redis.store[_LOCK_KEY] = "newer-token"
        release = mocker.spy(batch_outcome_mod, "release_dream_lock")

        await _deliver_sanitize()

        apply.assert_not_awaited()
        release.assert_not_called()
        await _assert_ended_without_applying(
            fake_dream_db, fake_dream_redis, LOCK_LOST_ERROR, gate_spent=False
        )
        assert fake_dream_redis.store[_LOCK_KEY] == "newer-token"
        row = fake_dream_db.rows["p1"]
        assert (row["status"], row["error"]) == (
            DreamPassStatus.ERRORED,
            LOCK_LOST_ERROR,
        )
        assert _charged(charges) == ["consolidate", "recombine", "sanitize"]

    @pytest.mark.parametrize("failure", ["timeout", "error"])
    async def test_a_fence_renewal_without_an_answer_never_applies_after_its_replacement(
        self,
        fake_dream_db,
        fake_dream_redis,
        apply,
        charges,
        sync_pass,
        mocker,
        failure,
    ):
        """Codex's failed-final-renewal case, the timeout at the real two
        seconds: the last callback claims its gate, and its compare-and-extend
        gets no answer while its lock lapses and a forced admin pass takes the
        scope and applies. Ownership unknown is not ownership: the callback
        ends without applying."""
        await _seed_batch_pass(fake_dream_db, fake_dream_redis)
        extend = lease_mod.extend_dream_lock
        at_fence, proceed = asyncio.Event(), asyncio.Event()
        renewals = 0

        async def no_answer_at_the_fence(scope, token: str, ttl_seconds: int) -> bool:
            nonlocal renewals
            renewals += 1
            if renewals == 2:  # the landing renewal first, then the fence
                at_fence.set()
                await proceed.wait()
                if failure == "timeout":
                    await asyncio.Event().wait()
                raise ConnectionError("redis did not answer")
            return await extend(scope, token, ttl_seconds)

        mocker.patch.object(lease_mod, "extend_dream_lock", no_answer_at_the_fence)
        loop = asyncio.get_running_loop()
        started = loop.time()
        old = asyncio.create_task(_deliver_sanitize())
        await asyncio.wait_for(at_fence.wait(), 5)
        await fake_dream_redis.delete(_LOCK_KEY)

        newer = await orchestrator_mod.execute_dream_pass(
            "u1", trigger="admin", force=True
        )
        proceed.set()
        await asyncio.wait_for(old, 5)

        if failure == "timeout":
            assert 1.9 < loop.time() - started < 4
        assert newer.error is None and not newer.skipped
        assert [call.args[1] for call in apply.await_args_list] == [newer.pass_id]
        error = f"expired: forced by an admin-triggered dream pass {newer.pass_id}"
        await _assert_ended_without_applying(fake_dream_db, fake_dream_redis, error)
        assert _charged(charges) == ["consolidate", "recombine", "sanitize"]

    async def test_a_fence_renewal_without_an_answer_fails_closed_and_releases(
        self, fake_dream_db, fake_dream_redis, apply, charges, mocker
    ):
        """With no newer pass about: the pass does not apply on a lock it
        cannot prove, ends with the lease it could not renew, and releases
        the lock by compare-and-delete, which only ever deletes its own."""
        await _seed_batch_pass(fake_dream_db, fake_dream_redis)
        extend = lease_mod.extend_dream_lock
        renewals = 0

        async def no_answer_at_the_fence(scope, token: str, ttl_seconds: int) -> bool:
            nonlocal renewals
            renewals += 1
            if renewals == 2:
                raise TimeoutError()
            return await extend(scope, token, ttl_seconds)

        mocker.patch.object(lease_mod, "extend_dream_lock", no_answer_at_the_fence)

        await _deliver_sanitize()

        apply.assert_not_awaited()
        await _assert_ended_without_applying(
            fake_dream_db, fake_dream_redis, "apply: dream lease could not be renewed"
        )
        assert _LOCK_KEY not in fake_dream_redis.store

    async def test_a_cancel_after_the_last_check_is_too_late_and_applies_once(
        self, fake_dream_db, fake_dream_redis, apply, charges, mocker
    ):
        """The documented window: a cancel that lands after the pass's last
        check, while it claims its gate, is like one landing during apply.
        The pass still holds its lock, so it applies, once, and nothing
        else does; the row keeps CANCELLED."""
        await _seed_batch_pass(fake_dream_db, fake_dream_redis)
        claim = batch_callbacks_mod.claim_apply_gate

        async def cancelled_while_claiming(pass_id: str):
            assert (
                await cancel_dream_pass(pass_id, user_id="u1", reason="late")
            ).cancelled
            return await claim(pass_id)

        mocker.patch.object(
            batch_callbacks_mod, "claim_apply_gate", cancelled_while_claiming
        )

        await _deliver_sanitize()

        assert [call.args[1] for call in apply.await_args_list] == ["p1"]
        row = fake_dream_db.rows["p1"]
        assert (row["status"], row["cancel_generation"]) == (
            DreamPassStatus.CANCELLED,
            1,
        )
        status = await job_status.read_status(kind="dream_pass", job_id="j1")
        assert status is not None and status.state == "complete"
        assert _LOCK_KEY not in fake_dream_redis.store


class TestTheSyncRoute:
    async def test_a_pass_whose_lock_lapsed_before_apply_does_not_apply(
        self, fake_dream_db, fake_dream_redis, apply, sync_pass, mocker
    ):
        """The lock lapsed during sanitize and a newer pass took it: the pass
        ends before apply with its phases' usage, and the newer holder's lock
        stays."""
        await _run_with_lock_taken_during_sanitize(mocker, fake_dream_redis)
        result = await orchestrator_mod.execute_dream_pass("u1")

        assert result.error == LOCK_LOST_ERROR
        assert result.usage is not None
        assert [p.phase for p in result.usage.phases] == [
            "consolidate",
            "recombine",
            "sanitize",
        ]
        apply.assert_not_awaited()
        assert fake_dream_redis.store[_LOCK_KEY] == "newer-token"
        row = fake_dream_db.rows[result.pass_id]
        assert (row["status"], row["error"]) == (
            DreamPassStatus.ERRORED,
            LOCK_LOST_ERROR,
        )

    async def test_a_pass_that_cannot_read_its_lock_before_apply_does_not_apply(
        self, fake_dream_db, fake_dream_redis, apply, sync_pass, mocker
    ):
        mocker.patch.object(
            locks_mod.DreamLockHandle,
            "held",
            AsyncMock(side_effect=ConnectionError("redis down")),
        )

        result = await orchestrator_mod.execute_dream_pass("u1")

        assert result.error == LOCK_LOST_ERROR
        apply.assert_not_awaited()
        assert _LOCK_KEY not in fake_dream_redis.store

    async def test_a_pass_that_still_holds_its_lock_applies(
        self, fake_dream_db, fake_dream_redis, apply, sync_pass
    ):
        result = await orchestrator_mod.execute_dream_pass("u1")

        assert result.error is None
        assert [call.args[1] for call in apply.await_args_list] == [result.pass_id]


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
        content="I ship on Fridays",
        source_description=None,
        valid_at=None,
        created_at=None,
    )
    return DreamInput(
        user_id="u1",
        group_id=_SCOPE.group_id,
        window_start=now - timedelta(days=14),
        window_end=now,
        episodes=[episode],
    )


async def _run_with_lock_taken_during_sanitize(mocker, fake_dream_redis) -> None:
    """Make the sanitize call hand the scope's lock to a newer pass."""
    answers = iter((ConsolidationOutput(), RecombinationOutput(), _OPS))

    async def answer(*_args, **_kwargs) -> StructuredCompletion:
        value = next(answers)
        if value is _OPS:
            fake_dream_redis.store[_LOCK_KEY] = "newer-token"
        return StructuredCompletion(value=value, usage=_BILLED)

    mocker.patch.object(
        orchestrator_mod, "structured_complete", AsyncMock(side_effect=answer)
    )


async def _seed_batch_pass(fake_dream_db, fake_dream_redis) -> None:
    """Batch pass p1 at its last phase: its lock held under the token its
    bundle carries, consolidate and recombine landed, its job and row open."""
    now = datetime.now(timezone.utc)
    await persist_input_bundle(
        "p1",
        DreamInput(
            user_id="u1", group_id=_SCOPE.group_id, window_start=now, window_end=now
        ),
        lock_token="our-token",
    )
    fake_dream_redis.store[_LOCK_KEY] = "our-token"
    await job_status.write_initial_status(kind="dream_pass", job_id="j1", user_id="u1")
    landed: tuple[tuple[DreamPhase, BaseModel], ...] = (
        ("consolidate", ConsolidationOutput()),
        ("recombine", RecombinationOutput()),
    )
    for phase, output in landed:
        await write_phase_to_state(
            pass_id="p1", phase=phase, row=_landed(phase, output)
        )
    fake_dream_db.seed(
        DreamPassDraft(
            id="p1",
            user_id="u1",
            scope_key=_SCOPE.scope_key,
            route=DreamPassRoute.ANTHROPIC_BATCH,
            trigger=DreamPassTrigger.CRON,
            status=DreamPassStatus.SUBMITTED,
            phase=DreamPassPhase.SANITIZE,
        ),
        provider_batch_id="msgbatch_sanitize",
        lease_token="our-token",
    )


async def _deliver_sanitize() -> None:
    now = datetime.now(timezone.utc)
    entry = PendingEntry(
        provider="anthropic",
        provider_batch_id="msgbatch_sanitize",
        callback_namespace="dream_pass",
        submitted_at=now,
        next_poll_at=now,
        payload={
            "user_id": "u1",
            "pass_id": "p1",
            "job_id": "j1",
            "phase": "sanitize",
            "phase_models": _PHASE_MODELS,
        },
    )
    await handle_dream_batch_result(entry, [_landed("sanitize", _OPS)])


def _landed(phase: str, output: BaseModel) -> BatchResultRow:
    return BatchResultRow(
        custom_id=f"p1:{phase}",
        content=output.model_dump_json(),
        input_tokens=10,
        output_tokens=20,
    )


async def _assert_ended_without_applying(
    fake_dream_db, fake_dream_redis, error: str, *, gate_spent: bool = True
) -> None:
    """Ended as ``fail_pass`` ends a pass: its job errored with *error*, its
    batch state and bundle cleaned, its apply gate spent when it got that
    far (else never claimed)."""
    status = await job_status.read_status(kind="dream_pass", job_id="j1")
    assert status is not None and (status.state, status.error) == ("errored", error)
    assert state_key("p1") not in fake_dream_redis.hashes
    assert input_bundle_key("p1") not in fake_dream_redis.store
    assert ("dream:applied:p1" in fake_dream_redis.store) is gate_spent
    assert DreamPassStatus.COMPLETE not in fake_dream_db.statuses("p1")


def _charged(charges: AsyncMock) -> list[str]:
    return [call.args[0].job.phase for call in charges.await_args_list]
