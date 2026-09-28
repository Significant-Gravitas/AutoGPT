"""The recall guard on both routes, end to end: a pass whose input shows one
fact recalled (dropped at clamp time) and whose apply finds another recalled
since the gather (dropped when apply reads the stamps again) demotes neither,
and its ``protected_demotions`` reaches the result, the durable record and the
admin job status. Only the LLM, the graph writes and the chat store are
stubbed; the lock, lease, record and job status run on the in-memory Redis and
DreamPass store from ``conftest.py``."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)

from backend.copilot.graphiti.recall_stamp import RecallStamp, stamp_time
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.inference.complete import StructuredCompletion
from backend.copilot.inference.context import InferenceUsage, RouteDecision
from backend.data.dream_pass_models import DreamPassDraft
from backend.executor.batch_executor import PendingEntry
from backend.executor.scheduler import execute_dream_pass_with_status
from backend.util.llm.providers import BatchResultRow

from . import apply as apply_mod
from . import batch_callbacks as batch_callbacks_mod
from . import orchestrator as orchestrator_mod
from . import recall_guard
from .batch_callbacks import handle_dream_batch_result
from .batch_submit import persist_input_bundle
from .fetch import DreamInput, FactRow
from .job_status import read_status, write_initial_status
from .pass_record import dream_pass_result_from_row
from .schemas import (
    ConsolidationOutput,
    DreamDemotion,
    DreamOperations,
    DreamPassResult,
    RecombinationOutput,
)


def _ago(days: float) -> str:
    return stamp_time(datetime.now(timezone.utc) - timedelta(days=days))


def _fact(uuid: str, last_recalled_at: str | None = None) -> FactRow:
    return FactRow(
        uuid=uuid,
        source="Nick",
        target="Atlas",
        name="works_on",
        fact=f"fact {uuid}",
        scope="real:global",
        confidence=0.8,
        status="active",
        created_at="2026-01-01T00:00:00+00:00",
        recall_count=1 if last_recalled_at else None,
        last_recalled_at=last_recalled_at,
    )


def _bundle(user_id: str) -> DreamInput:
    """``held`` was recalled two days before the gather, ``late`` 40 days
    before it (outside the window), ``cold`` never; 100 facts, a cap of 5."""
    facts = [
        _fact("held", _ago(2)),
        _fact("late", _ago(40)),
        _fact("cold"),
        *(_fact(f"f{i}") for i in range(97)),
    ]
    now = datetime.now(timezone.utc)
    return DreamInput(
        user_id=user_id,
        group_id=MemoryScope.for_user(user_id).group_id,
        window_start=now - timedelta(days=14),
        window_end=now,
        facts=facts,
        known_fact_uuids={f.uuid for f in facts},
    )


_SANITIZED = DreamOperations(
    demotions=[
        DreamDemotion(edge_uuid=uuid, reason="stale_fact")
        for uuid in ("held", "late", "cold")
    ],
    summary_for_user="Tidied up.",
)


@pytest.fixture
def graph(mocker) -> SimpleNamespace:
    """apply's graph writes and chat store; the guard's re-read finds ``late``
    recalled a moment ago, after the pass gathered its input."""
    state = SimpleNamespace(written=[])

    async def supersede(driver, uuids, **kwargs):
        state.written.extend(uuids)
        return list(uuids), []

    async def reread(driver, group_id, uuids):
        now = stamp_time(datetime.now(timezone.utc))
        return [
            RecallStamp(uuid=uuid, recall_count=2, last_recalled_at=now)
            for uuid in uuids
            if uuid == "late"
        ]

    driver = MagicMock()
    driver.close = AsyncMock()
    mocker.patch.object(apply_mod, "open_driver", return_value=driver)
    mocker.patch.object(apply_mod, "mark_edges_superseded", side_effect=supersede)
    mocker.patch.object(recall_guard, "read_recall_stamps", side_effect=reread)
    database = MagicMock()
    database.create_chat_session = AsyncMock()
    database.update_chat_session_title = AsyncMock()
    database.add_chat_message = AsyncMock()
    mocker.patch("backend.data.db_accessors.chat_db", return_value=database)
    mocker.patch(
        "backend.api.features.orgs.db.get_user_default_team",
        AsyncMock(return_value=(None, None)),
    )
    return state


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


@pytest.fixture
def scheduler_loop(mocker):
    """The scheduler's shared event loop, which only a started scheduler
    has: one loop of the test's own that its ``run_async`` runs on."""
    loop = asyncio.new_event_loop()
    mocker.patch(
        "backend.executor.scheduler.run_async",
        side_effect=lambda coro, timeout=None: loop.run_until_complete(coro),
    )
    yield loop
    loop.close()


def test_the_sync_route_counts_both_drops_everywhere(
    mocker, graph, fake_dream_db, scheduler_loop
) -> None:
    """Driven through the scheduler's own admin wrapper, which writes the
    pass's result to its job status."""
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
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_bundle("u-sync")),
    )
    mocker.patch.object(
        orchestrator_mod,
        "structured_complete",
        AsyncMock(
            side_effect=[
                _answer(ConsolidationOutput(facts=[])),
                _answer(RecombinationOutput(proposals=[])),
                _answer(_SANITIZED),
            ]
        ),
    )
    scheduler_loop.run_until_complete(
        write_initial_status(kind="dream_pass", job_id="j-sync", user_id="u")
    )

    execute_dream_pass_with_status("u-sync", "j-sync")

    job = scheduler_loop.run_until_complete(_job_result("j-sync"))
    assert graph.written == ["cold"]
    assert (job.error, job.demotion_count, job.protected_demotions) == (None, 1, 2)
    row = fake_dream_db.rows[job.pass_id]
    planned = row["operations"]["planned"]
    assert [d.edge_uuid for d in planned.demotions] == ["late", "cold"]
    assert row["operations"]["applied"].protected_demotions == 2
    record = dream_pass_result_from_row(fake_dream_db.record(job.pass_id))
    assert record.protected_demotions == 2


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
async def test_the_batch_route_rereads_the_stamps_hours_after_its_gather(
    mocker, graph, fake_dream_db, fake_dream_redis
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
        ("sanitize", json.dumps(_SANITIZED.model_dump())),
    ):
        await handle_dream_batch_result(_entry(phase), [_row(phase, content)])

    assert graph.written == ["cold"]
    row = fake_dream_db.rows["p-batch"]
    assert row["status"] is DreamPassStatus.COMPLETE
    assert row["operations"]["applied"].protected_demotions == 2
    result = dream_pass_result_from_row(fake_dream_db.record("p-batch"))
    assert (result.demotion_count, result.protected_demotions) == (1, 2)
    assert (await _job_result("j-batch")).protected_demotions == 2
