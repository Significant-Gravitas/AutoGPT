"""The renewals that admit a dream pass's writes, on both routes, through the
real orchestrator, callbacks and apply body over the in-memory Redis and
store: a final renewal that gets no answer is not ownership, a renewal whose
answer arrives after the lock lapsed is caught by apply's own renewal before
its first write, and so is a lock that lapses while apply creates its
session. In each, a newer forced pass takes the scope and writes; the old
pass writes nothing. Only the models, the budget and apply's graph and chat
writes are stubbed (the writes recorded per pass)."""

import asyncio
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
from backend.copilot.inference.complete import StructuredCompletion
from backend.copilot.inference.context import InferenceUsage, RouteDecision
from backend.data.dream_pass_models import DreamPassDraft
from backend.executor.batch_executor import PendingEntry
from backend.util.llm.providers import BatchResultRow

from . import apply as apply_mod
from . import job_status
from . import lease as lease_mod
from . import orchestrator as orchestrator_mod
from .batch_callbacks import handle_dream_batch_result
from .batch_state import write_phase_to_state
from .batch_submit import persist_input_bundle
from .fetch import DreamInput, EpisodeRow
from .locks import BATCH_LOCK_TTL_SECONDS
from .schemas import (
    ConsolidatedFact,
    ConsolidationOutput,
    DreamOperations,
    DreamPhase,
    RecombinationOutput,
)

_SCOPE = MemoryScope.for_user("u1")
_LOCK_KEY = _SCOPE.redis_key("dream_lock")
_FACT = ConsolidatedFact(content="Nick ships on Fridays", confidence=0.9)
_OPS = DreamOperations(writes=[_FACT], summary_for_user="ok")
_ANSWERS: tuple[BaseModel, ...] = (ConsolidationOutput(), RecombinationOutput(), _OPS)
_BILLED = InferenceUsage(
    model="m", input_tokens=100, output_tokens=20, payer="platform_allowance"
)


class Graph:
    """What apply wrote, one pass id per fact; a pass named in ``hold`` stops
    in its session creation until ``resume`` is set."""

    def __init__(self) -> None:
        self.writes: list[str] = []
        self.hold: set[str] = set()
        self.in_session = asyncio.Event()
        self.resume = asyncio.Event()


@pytest.fixture
def graph(mocker) -> Graph:
    recorded = Graph()

    async def create_session(scope, pass_id: str) -> str:
        if pass_id in recorded.hold:
            recorded.in_session.set()
            await recorded.resume.wait()
        return f"s-{pass_id}"

    async def write_fact(scope, pass_id: str, index, fact, **_kwargs) -> bool:
        recorded.writes.append(pass_id)
        return False

    mocker.patch.object(apply_mod, "_create_dream_session", create_session)
    mocker.patch.object(apply_mod, "_write_consolidated_fact", write_fact)
    mocker.patch.object(apply_mod, "_write_dream_summary_message", AsyncMock())
    return recorded


@pytest.fixture(autouse=True)
def sync_pass(mocker) -> None:
    """A sync pass for the scope, its three phases answering in turn, pass
    after pass."""
    calls = 0

    async def answer(*_args, **_kwargs) -> StructuredCompletion:
        nonlocal calls
        calls += 1
        return StructuredCompletion(value=_ANSWERS[(calls - 1) % 3], usage=_BILLED)

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
    mocker.patch.object(orchestrator_mod, "structured_complete", side_effect=answer)
    mocker.patch("backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock())


class TestTheSyncRoute:
    @pytest.mark.parametrize("failure", ["timeout", "error"])
    async def test_a_final_renewal_without_an_answer_never_applies_after_its_replacement(
        self, monkeypatch, fake_dream_db, fake_dream_redis, graph, failure
    ):
        """Codex's reproduction, the timeout at the real two seconds: the
        pass's renewal before apply gets no answer while its lock lapses and
        a forced admin pass takes the scope and writes. The old pass ends
        there, having written nothing."""
        at_renewal, proceed = asyncio.Event(), asyncio.Event()
        evaluate = fake_dream_redis.eval
        tokens: list[str] = []

        async def no_answer_before_apply(script: str, numkeys: int, *args):
            if '"expire"' in script:
                tokens.append(args[1])
                if tokens.count(tokens[0]) == 4 and args[1] == tokens[0]:
                    at_renewal.set()
                    await proceed.wait()
                    if failure == "timeout":
                        await asyncio.Event().wait()
                    raise ConnectionError("redis did not answer")
            return await evaluate(script, numkeys, *args)

        monkeypatch.setattr(fake_dream_redis, "eval", no_answer_before_apply)
        loop = asyncio.get_running_loop()
        started = loop.time()
        old = asyncio.create_task(orchestrator_mod.execute_dream_pass("u1"))
        await asyncio.wait_for(at_renewal.wait(), 5)
        await fake_dream_redis.delete(_LOCK_KEY)

        newer = await orchestrator_mod.execute_dream_pass(
            "u1", trigger="admin", force=True
        )
        proceed.set()
        result = await asyncio.wait_for(old, 5)

        if failure == "timeout":
            assert 1.9 < loop.time() - started < 4
        assert newer.error is None and not newer.skipped
        assert graph.writes == [newer.pass_id]
        assert result.error == (
            f"expired: forced by an admin-triggered dream pass {newer.pass_id}"
        )
        assert fake_dream_db.rows[result.pass_id]["status"] is DreamPassStatus.EXPIRED

    async def test_a_lock_that_lapses_while_apply_starts_stops_its_writes(
        self, fake_dream_db, fake_dream_redis, graph
    ):
        """The pass is admitted and enters apply; its lock lapses while the
        session is created, and a forced admin pass takes the scope and
        writes. Apply's renewal before its first write finds the lock gone:
        the old pass writes nothing."""
        old = asyncio.create_task(orchestrator_mod.execute_dream_pass("u1"))
        await _hold_the_first_session(graph, fake_dream_db)
        await fake_dream_redis.delete(_LOCK_KEY)

        newer = await orchestrator_mod.execute_dream_pass(
            "u1", trigger="admin", force=True
        )
        graph.resume.set()
        result = await asyncio.wait_for(old, 5)

        assert newer.error is None
        assert graph.writes == [newer.pass_id]
        assert result.error is not None and result.error.startswith("Dream lock lost")


class TestTheBatchRoute:
    async def test_a_lock_that_lapses_while_apply_starts_stops_its_writes(
        self, fake_dream_db, fake_dream_redis, graph
    ):
        """Codex's long-apply model: the last callback passes its fence and
        enters apply; a whole batch lease lapses while the session is
        created, and a forced admin pass takes the scope and writes. Apply's
        renewal before its first write finds the lock gone."""
        await _seed_batch_pass(fake_dream_db, fake_dream_redis)
        graph.hold.add("p1")
        old = asyncio.create_task(_deliver_sanitize())
        await asyncio.wait_for(graph.in_session.wait(), 5)
        await fake_dream_redis.delete(_LOCK_KEY)

        newer = await orchestrator_mod.execute_dream_pass(
            "u1", trigger="admin", force=True
        )
        graph.resume.set()
        await asyncio.wait_for(old, 5)

        assert newer.error is None
        assert graph.writes == [newer.pass_id]
        status = await job_status.read_status(kind="dream_pass", job_id="j1")
        assert status is not None and status.state == "errored"
        assert (status.error or "").startswith("apply: DreamLockLostError")

    async def test_a_fence_answer_held_past_the_lease_still_writes_nothing(
        self, mocker, fake_dream_db, fake_dream_redis, graph
    ):
        """PR 7's delayed-reply case on the renewal that replaced the GET:
        the fence's compare-and-extend succeeds, its answer held while the
        lock lapses (a stall past the TTL) and a forced admin pass takes the
        scope and writes. The stale "yours" admits the old callback to apply,
        and apply's renewal before its first write finds the lock gone."""
        await _seed_batch_pass(fake_dream_db, fake_dream_redis)
        extend = lease_mod.extend_dream_lock
        answered, deliver = asyncio.Event(), asyncio.Event()
        renewals = 0

        async def held_at_the_fence(scope, token: str, ttl_seconds: int) -> bool:
            nonlocal renewals
            renewals += 1
            renewed = await extend(scope, token, ttl_seconds)
            if renewals == 2:
                answered.set()
                await deliver.wait()
            return renewed

        mocker.patch.object(lease_mod, "extend_dream_lock", held_at_the_fence)
        old = asyncio.create_task(_deliver_sanitize())
        await asyncio.wait_for(answered.wait(), 5)
        assert fake_dream_redis.ttls[_LOCK_KEY] == BATCH_LOCK_TTL_SECONDS
        await fake_dream_redis.delete(_LOCK_KEY)

        newer = await orchestrator_mod.execute_dream_pass(
            "u1", trigger="admin", force=True
        )
        deliver.set()
        await asyncio.wait_for(old, 5)

        assert newer.error is None
        assert graph.writes == [newer.pass_id]

    async def test_a_pass_that_keeps_its_lock_writes_once(
        self, fake_dream_db, fake_dream_redis, graph
    ):
        await _seed_batch_pass(fake_dream_db, fake_dream_redis)

        await _deliver_sanitize()

        assert graph.writes == ["p1"]
        status = await job_status.read_status(kind="dream_pass", job_id="j1")
        assert status is not None and status.state == "complete"
        assert _LOCK_KEY not in fake_dream_redis.store


async def _hold_the_first_session(graph: Graph, fake_dream_db) -> None:
    """Hold the first pass in its session creation, once it is there."""
    while not fake_dream_db.rows:
        await asyncio.sleep(0)
    graph.hold.add(next(iter(fake_dream_db.rows)))
    await asyncio.wait_for(graph.in_session.wait(), 5)


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
        await write_phase_to_state(pass_id="p1", phase=phase, row=_row(phase, output))
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
            "phase_models": {
                p: "claude-sonnet-5" for p in ("consolidate", "recombine", "sanitize")
            },
        },
    )
    await handle_dream_batch_result(entry, [_row("sanitize", _OPS)])


def _row(phase: str, output: BaseModel) -> BatchResultRow:
    return BatchResultRow(
        custom_id=f"p1:{phase}",
        content=output.model_dump_json(),
        input_tokens=10,
        output_tokens=20,
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
        content="I ship on Fridays",
        source_description=None,
        valid_at=None,
        created_at=None,
    )
    return DreamInput(
        user_id="u1",
        group_id=_SCOPE.group_id,
        window_start=now,
        window_end=now,
        episodes=[episode],
    )
