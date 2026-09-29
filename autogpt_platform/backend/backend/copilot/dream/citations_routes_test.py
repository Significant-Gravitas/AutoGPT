"""Writes citing nothing a pass read, end to end on both routes: the sanitizer
lets through one consolidation citing a fact the pass read, one citing
nothing, and a proposal citing only a uuid the pass never read. The first is
queued with what it cites; the other two are dropped before they reach the
graph and reach ``uncited_writes_dropped`` in the result, the durable record
and the admin job status. The first also cites a fact of another scope,
which is dropped and counted in ``cross_scope_citations_dropped`` there.
The worker then drops the queued write for resting on a forget, fails
another and makes a third whose record fails: ``dropped_forgotten``,
``failed_writes`` and ``provenance_pending`` report them in the same places
on the sync route, which waits for the worker, and not on the batch route,
which does not. Only the LLM, the ingestion queue and worker, and the chat
store are stubbed; the rest runs on ``conftest.py``'s in-memory Redis and
store."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)

from backend.copilot.graphiti.ingest import IngestionCompletion
from backend.copilot.graphiti.recall_citations import Citations
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.inference.complete import StructuredCompletion
from backend.copilot.inference.context import InferenceUsage, RouteDecision
from backend.data.dream_pass_models import DreamPassApplied, DreamPassDraft
from backend.executor.batch_executor import PendingEntry
from backend.executor.scheduler import execute_dream_pass_with_status
from backend.util.llm.providers import BatchResultRow

from . import apply as apply_mod
from . import batch_callbacks as batch_callbacks_mod
from . import orchestrator as orchestrator_mod
from .batch_callbacks import handle_dream_batch_result
from .batch_submit import persist_input_bundle
from .fetch import DreamInput, FactRow
from .job_status import read_status, write_initial_status
from .pass_record import dream_pass_result_from_row
from .schemas import (
    ConsolidatedFact,
    ConsolidationOutput,
    DreamOperations,
    DreamPassResult,
    ProposedFinding,
    RecombinationOutput,
)

_OPS = DreamOperations(
    writes=[
        ConsolidatedFact(
            content="Nick ships on Fridays",
            confidence=0.9,
            source_fact_uuids=["f-read", "f-project"],
        ),
        ConsolidatedFact(content="Nick ships on Mondays", confidence=0.9),
    ],
    proposals=[
        ProposedFinding(
            content="Nick prefers small releases",
            confidence=0.5,
            rationale="a hunch",
            source_fact_uuids=["f-never-read"],
        )
    ],
    summary_for_user="Consolidated a fact.",
)


def _bundle(user_id: str) -> DreamInput:
    now = datetime.now(timezone.utc)
    fact = FactRow(
        uuid="f-read",
        source="Nick",
        target="Fridays",
        name="ships_on",
        fact="Nick ships on Fridays",
        scope="real:global",
        confidence=0.8,
        status="active",
        created_at="2026-01-01T00:00:00+00:00",
    )
    elsewhere = fact.model_copy(update={"uuid": "f-project", "scope": "project:x"})
    return DreamInput(
        user_id=user_id,
        group_id=MemoryScope.for_user(user_id).group_id,
        window_start=now - timedelta(days=14),
        window_end=now,
        facts=[fact, elsewhere],
        known_fact_uuids={fact.uuid, elsewhere.uuid},
    )


async def _worker_reports(completion: IngestionCompletion, _: float) -> bool:
    """The worker's outcomes, one of each, as its counters carry them: a
    write dropped for resting on a forget, one it failed to make, one made
    whose record is pending."""
    completion.dropped_forgotten += 1
    completion.failed += 1
    completion.provenance_pending += 1
    return True


@pytest.fixture
def queued(mocker) -> AsyncMock:
    """The ingestion queue and its worker's outcomes; apply's chat store."""
    enqueue = AsyncMock(return_value=True)
    mocker.patch.object(apply_mod, "enqueue_episode", enqueue)
    mocker.patch.object(apply_mod, "wait_for_ingestion", _worker_reports)
    database = MagicMock()
    database.create_chat_session = AsyncMock()
    database.update_chat_session_title = AsyncMock()
    database.add_chat_message = AsyncMock()
    mocker.patch("backend.data.db_accessors.chat_db", return_value=database)
    mocker.patch(
        "backend.api.features.orgs.db.get_user_default_team",
        AsyncMock(return_value=(None, None)),
    )
    return enqueue


def _answer(value) -> StructuredCompletion:
    return StructuredCompletion(
        value=value, usage=InferenceUsage(model="m", payer="platform_allowance")
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


async def _job_result(job_id: str) -> DreamPassResult:
    status = await read_status(kind="dream_pass", job_id=job_id)
    assert status is not None and status.state == "complete"
    return DreamPassResult.model_validate(status.result)


def _reported(result: DreamPassResult | DreamPassApplied) -> tuple[int, ...]:
    """Written, proposed, uncited, cross-scope citations, and the worker's
    three outcomes: dropped as forgotten, failed, provenance pending."""
    return (
        result.consolidated_count,
        result.proposal_count,
        result.uncited_writes_dropped,
        result.cross_scope_citations_dropped,
        result.dropped_forgotten,
        result.failed_writes,
        result.provenance_pending,
    )


def _only_the_cited_write_was_queued(enqueue: AsyncMock) -> None:
    [call] = enqueue.await_args_list
    assert call.kwargs["citations"] == Citations(fact_uuids=["f-read"])


@pytest.fixture
def scheduler_loop(mocker):
    """The scheduler's shared event loop: one loop of the test's own that
    its ``run_async`` runs on."""
    loop = asyncio.new_event_loop()
    mocker.patch(
        "backend.executor.scheduler.run_async",
        side_effect=lambda coro, timeout=None: loop.run_until_complete(coro),
    )
    yield loop
    loop.close()


def test_the_sync_route_reports_the_dropped_writes_everywhere(
    mocker, queued, fake_dream_db, scheduler_loop
) -> None:
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
        "gather_dream_input",
        AsyncMock(return_value=_bundle("u-sync")),
    )
    mocker.patch.object(
        orchestrator_mod,
        "record_phase_cost",
        AsyncMock(side_effect=lambda ctx, usage: usage),
    )
    mocker.patch.object(
        orchestrator_mod,
        "structured_complete",
        AsyncMock(
            side_effect=[
                _answer(ConsolidationOutput(facts=[])),
                _answer(RecombinationOutput(proposals=[])),
                _answer(_OPS),
            ]
        ),
    )
    scheduler_loop.run_until_complete(
        write_initial_status(kind="dream_pass", job_id="j-sync", user_id="u")
    )

    execute_dream_pass_with_status("u-sync", "j-sync")

    job = scheduler_loop.run_until_complete(_job_result("j-sync"))
    assert job.error is None
    assert _reported(job) == (1, 0, 2, 1, 1, 1, 1)
    applied = fake_dream_db.rows[job.pass_id]["operations"]["applied"]
    assert _reported(applied) == (1, 0, 2, 1, 1, 1, 1)
    record = dream_pass_result_from_row(fake_dream_db.record(job.pass_id))
    assert _reported(record) == (1, 0, 2, 1, 1, 1, 1)
    _only_the_cited_write_was_queued(queued)


def _entry(phase: str) -> PendingEntry:
    now = datetime.now(timezone.utc)
    custom_id = f"p-batch:{phase}"
    return PendingEntry(
        provider="anthropic",
        provider_batch_id=f"msgbatch_{phase}",
        callback_namespace="dream_pass",
        submitted_at=now,
        next_poll_at=now,
        payload={
            "user_id": "u-batch",
            "pass_id": "p-batch",
            "job_id": "j-batch",
            "phase": phase,
            "phase_models": {
                "consolidate": "claude-sonnet-5",
                "recombine": "claude-opus-5-5",
                "sanitize": "claude-sonnet-5",
            },
            "custom_ids": [custom_id],
            "phase_for_custom_id": {custom_id: phase},
        },
    )


def _row(phase: str, content: str) -> BatchResultRow:
    return BatchResultRow(
        custom_id=f"p-batch:{phase}", content=content, input_tokens=10, output_tokens=20
    )


@pytest.mark.asyncio
async def test_the_batch_route_reports_the_dropped_writes_everywhere(
    mocker, queued, fake_dream_db, fake_dream_redis
) -> None:
    scope = MemoryScope.for_user("u-batch")
    fake_dream_redis.store[scope.redis_key("dream_lock")] = "tok-batch"
    await persist_input_bundle("p-batch", _bundle("u-batch"), lock_token="tok-batch")
    fake_dream_db.seed(
        DreamPassDraft(
            id="p-batch",
            user_id="u-batch",
            scope_key=scope.scope_key,
            route=DreamPassRoute.ANTHROPIC_BATCH,
            trigger=DreamPassTrigger.CRON,
            status=DreamPassStatus.SUBMITTED,
            phase=DreamPassPhase.CONSOLIDATE,
            started_at=datetime.now(timezone.utc),
        )
    )
    await write_initial_status(kind="dream_pass", job_id="j-batch", user_id="u-batch")
    mocker.patch.object(
        batch_callbacks_mod,
        "submit_phase",
        AsyncMock(
            side_effect=[
                MagicMock(provider_batch_id="msgbatch_recombine"),
                MagicMock(provider_batch_id="msgbatch_sanitize"),
            ]
        ),
    )
    mocker.patch.object(batch_callbacks_mod, "anthropic_api_key", return_value="sk")
    mocker.patch(
        "backend.copilot.inference.record.persist_and_record_usage", AsyncMock()
    )

    for phase, content in (
        ("consolidate", '{"facts": []}'),
        ("recombine", '{"proposals": []}'),
        ("sanitize", json.dumps(_OPS.model_dump())),
    ):
        await handle_dream_batch_result(_entry(phase), [_row(phase, content)])

    row = fake_dream_db.rows["p-batch"]
    assert row["status"] is DreamPassStatus.COMPLETE
    assert _reported(row["operations"]["applied"]) == (1, 0, 2, 1, 0, 0, 0)
    record = dream_pass_result_from_row(fake_dream_db.record("p-batch"))
    assert _reported(record) == (1, 0, 2, 1, 0, 0, 0)
    assert _reported(await _job_result("j-batch")) == (1, 0, 2, 1, 0, 0, 0)
    _only_the_cited_write_was_queued(queued)
