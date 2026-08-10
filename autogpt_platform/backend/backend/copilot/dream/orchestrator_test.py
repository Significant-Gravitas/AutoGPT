"""Orchestrator three-phase tests with the LLM + Graphiti calls mocked.

The orchestrator is the integration seam — these tests exercise the
phase-to-phase plumbing without actually hitting OpenRouter or
FalkorDB. apply + fetch get their own integration tests; this file is
the unit-level safety net for the control-flow.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.executor.scheduler import SCHEDULER_DREAM_OPERATION_TIMEOUT_SECONDS
from backend.util.llm.providers import DEFAULT_REQUEST_TIMEOUT_SECONDS

from . import orchestrator as orchestrator_mod
from .apply import INGESTION_DRAIN_TIMEOUT_SECONDS, LOCK_DRAIN_RENEWAL_SECONDS
from .fetch import DreamInput, EpisodeRow, FactRow
from .llm import CompletionUsage, DreamLLMError, StructuredCompletion
from .locks import DEFAULT_LOCK_TTL_SECONDS
from .schemas import (
    ConsolidatedFact,
    ConsolidationOutput,
    DreamDemotion,
    DreamOperations,
    EntityInvalidation,
    IngestionDrainStatus,
    ProposedFinding,
    RecombinationOutput,
)


def _wrap(value, model: str = "test-model") -> StructuredCompletion:
    """Wrap a phase output in StructuredCompletion with zeroed usage.

    Tests that don't care about token bookkeeping use this so they can
    keep the side_effect list short. Tests that exercise the usage
    pipeline build their own ``CompletionUsage`` with real numbers.
    """
    return StructuredCompletion(value=value, usage=CompletionUsage(model=model))


@pytest.fixture(autouse=True)
def force_sync_baseline(mocker):
    """Pin the orchestrator to the sync_baseline path for every test
    in this file.

    Step 5 of the plan routes dream pass to the Anthropic batch path
    when an Anthropic key is configured; these tests mock
    ``structured_completion`` directly to exercise the sync three-phase
    flow, so we have to override the routing decision to keep them
    valid. The batch path has its own dedicated test coverage in
    ``batch_callbacks_test.py``.
    """
    mocker.patch.object(
        orchestrator_mod,
        "resolve_dream_execution_path",
        return_value="sync_baseline",
    )


def _build_input(*, episodes=1, facts=1) -> DreamInput:
    return DreamInput(
        user_id="u",
        group_id="g",
        window_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 14, tzinfo=timezone.utc),
        episodes=[
            EpisodeRow(
                uuid=f"e{i}",
                name=None,
                content="hello",
                source_description=None,
                valid_at=None,
                created_at=None,
            )
            for i in range(episodes)
        ],
        facts=[
            FactRow(
                uuid=f"f{i}",
                source="A",
                target="B",
                name="likes",
                fact="A likes B",
                scope="real:global",
                confidence=0.7,
                status="active",
                created_at=None,
            )
            for i in range(facts)
        ],
        recent_sessions=[],
        known_fact_uuids={f"f{i}" for i in range(facts)},
        known_episode_uuids={f"e{i}" for i in range(episodes)},
    )


@asynccontextmanager
async def _noop_lock(*args, **kwargs):
    yield


@pytest.fixture(autouse=True)
def _stub_lock(mocker):
    """Always-acquired lock; tests that exercise the lock-held branch
    re-patch to a function that raises DreamLockHeld."""
    mocker.patch.object(orchestrator_mod, "dream_lock", _noop_lock)


@pytest.fixture(autouse=True)
def _stub_billing(mocker):
    """Default-allow billing so happy-path tests don't have to plumb
    Redis/Supabase. Tests that exercise the budget-skip path re-patch
    ``check_dream_budget`` with a (False, reason) AsyncMock.

    ``record_phase_cost`` is a no-op fire-and-forget here; the billing
    seam itself has dedicated coverage in ``billing_test.py``."""
    mocker.patch.object(
        orchestrator_mod, "check_dream_budget", AsyncMock(return_value=(True, None))
    )
    mocker.patch.object(orchestrator_mod, "record_phase_cost", AsyncMock())


@pytest.fixture(autouse=True)
def _stub_batch_flag(mocker):
    """Default the batch-path LD flag off so tests stay on sync + hermetic
    (no LaunchDarkly calls). The wiring test re-patches to assert flag flow."""
    mocker.patch.object(
        orchestrator_mod, "is_feature_enabled", AsyncMock(return_value=False)
    )


@pytest.fixture(autouse=True)
def _stub_marker_redis(mocker):
    """No last-completed marker by default (get → None) so every existing
    test runs the full pass; the stamp write is a silent success. The
    no-new-activity tests reconfigure ``get`` on the returned mock."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    mocker.patch(
        "backend.data.redis_client.get_redis_async",
        AsyncMock(return_value=redis),
    )
    return redis


@pytest.mark.asyncio
async def test_empty_input_returns_skipped(mocker):
    """No episodes AND no facts ⇒ skipped, no LLM calls."""
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_build_input(episodes=0, facts=0)),
    )
    structured = mocker.patch.object(
        orchestrator_mod,
        "structured_completion",
        AsyncMock(),
    )
    apply_mock = mocker.patch.object(orchestrator_mod, "apply_operations", AsyncMock())

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.skipped is True
    assert result.skip_reason == "no_input"
    structured.assert_not_called()
    apply_mock.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("flag_on", [True, False])
async def test_batch_path_gated_by_flag_not_key(mocker, flag_on):
    """The Anthropic batch path is gated by DREAM_PASS_BATCH_ENABLED — the
    flag value (not mere direct-key presence) is what flows to routing's
    ``batch_processing_enabled``, so the batch path can ship dark."""
    resolve = mocker.patch.object(
        orchestrator_mod,
        "resolve_dream_execution_path",
        return_value="sync_baseline",
    )
    flag = mocker.patch.object(
        orchestrator_mod,
        "is_feature_enabled",
        AsyncMock(return_value=flag_on),
    )
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_build_input(episodes=0, facts=0)),
    )

    await orchestrator_mod.execute_dream_pass("u")

    flag.assert_awaited_once()
    assert flag.await_args.args[0] is orchestrator_mod.Flag.DREAM_PASS_BATCH_ENABLED
    assert resolve.call_args.kwargs["batch_processing_enabled"] is flag_on


@pytest.mark.asyncio
async def test_happy_path_runs_three_steps_and_applies(mocker):
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_build_input()),
    )

    consolidated = ConsolidationOutput(
        facts=[ConsolidatedFact(content="A likes B", confidence=0.8)]
    )
    recombined = RecombinationOutput(
        proposals=[
            ProposedFinding(
                content="A probably trusts B",
                confidence=0.6,
                rationale="implied by A likes B",
            )
        ]
    )
    sanitized = DreamOperations(
        writes=consolidated.facts,
        proposals=recombined.proposals,
        demotions=[],
        entity_invalidations=[],
        summary_for_user="Dream consolidated 1 fact.",
    )
    mocker.patch.object(
        orchestrator_mod,
        "structured_completion",
        AsyncMock(
            side_effect=[_wrap(consolidated), _wrap(recombined), _wrap(sanitized)]
        ),
    )
    apply_mock = mocker.patch.object(
        orchestrator_mod,
        "apply_operations",
        AsyncMock(
            return_value={
                "session_id": "s1",
                "consolidated_count": 1,
                "proposal_count": 1,
                "demotion_count": 0,
                "demotion_failed_count": 0,
                "entity_invalidation_count": 0,
                # Non-default value so the assertion below proves the status
                # is threaded from apply_stats, not left at the schema default.
                "ingestion_drain_status": IngestionDrainStatus.timed_out,
            }
        ),
    )

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error is None
    assert result.skipped is False
    assert result.consolidated_count == 1
    assert result.proposal_count == 1
    assert result.summary_for_user == "Dream consolidated 1 fact."
    assert result.dream_session_id == "s1"
    assert result.ingestion_drain_status is IngestionDrainStatus.timed_out
    apply_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_held_dream_lock_handle_is_threaded_into_apply(mocker):
    """apply renews the dream lock before the drain + demotions, which it can
    only do with the handle the orchestrator holds. Dropping that kwarg would
    silently reinstate the lock-expiry-during-drain window, so pin it."""
    sentinel_handle = object()

    @asynccontextmanager
    async def _handle_lock(*args, **kwargs):
        yield sentinel_handle

    mocker.patch.object(orchestrator_mod, "dream_lock", _handle_lock)
    mocker.patch.object(
        orchestrator_mod, "gather_dream_input", AsyncMock(return_value=_build_input())
    )
    consolidated = ConsolidationOutput(
        facts=[ConsolidatedFact(content="A likes B", confidence=0.8)]
    )
    mocker.patch.object(
        orchestrator_mod,
        "structured_completion",
        AsyncMock(
            side_effect=[
                _wrap(consolidated),
                _wrap(RecombinationOutput(proposals=[])),
                _wrap(DreamOperations(writes=consolidated.facts)),
            ]
        ),
    )
    apply_mock = mocker.patch.object(
        orchestrator_mod,
        "apply_operations",
        AsyncMock(
            return_value={
                "session_id": "s1",
                "consolidated_count": 1,
                "ingestion_drain_status": IngestionDrainStatus.drained,
            }
        ),
    )

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error is None
    apply_mock.assert_awaited_once()
    assert apply_mock.await_args.kwargs["lock_handle"] is sentinel_handle


@pytest.mark.asyncio
async def test_missing_drain_key_folds_to_fail_closed_timed_out(mocker):
    """When apply_stats omits the drain key entirely (e.g. an upstream bug),
    the result must fail closed to ``timed_out`` — writes potentially at
    risk — never silently read as a confirmed ``drained`` success."""
    mocker.patch.object(
        orchestrator_mod, "gather_dream_input", AsyncMock(return_value=_build_input())
    )
    consolidated = ConsolidationOutput(
        facts=[ConsolidatedFact(content="A likes B", confidence=0.8)]
    )
    recombined = RecombinationOutput(proposals=[])
    sanitized = DreamOperations(summary_for_user="quiet night")
    mocker.patch.object(
        orchestrator_mod,
        "structured_completion",
        AsyncMock(
            side_effect=[_wrap(consolidated), _wrap(recombined), _wrap(sanitized)]
        ),
    )
    mocker.patch.object(
        orchestrator_mod,
        "apply_operations",
        # No ``ingestion_drain_status`` key at all.
        AsyncMock(return_value={"session_id": "s1", "consolidated_count": 1}),
    )

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error is None
    assert result.ingestion_drain_status is IngestionDrainStatus.timed_out


@pytest.mark.asyncio
async def test_each_phase_threads_its_own_llm_timeout_into_structured_completion(
    mocker,
):
    """Recombine/sanitize got 16384-token output budgets because real
    responses exceed 8192 tokens; at real decode speeds those responses
    outlive the shared 120s ``call_provider`` default, so each phase
    must hand ``structured_completion`` its own wall-clock budget —
    otherwise the timeout kills exactly the responses the token-cap
    raise was meant to save."""
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_build_input()),
    )
    consolidated = ConsolidationOutput(
        facts=[ConsolidatedFact(content="A likes B", confidence=0.8)]
    )
    recombined = RecombinationOutput(proposals=[])
    sanitized = DreamOperations(
        writes=[],
        proposals=[],
        demotions=[],
        entity_invalidations=[],
        summary_for_user="quiet night",
    )
    llm_mock = mocker.patch.object(
        orchestrator_mod,
        "structured_completion",
        AsyncMock(
            side_effect=[_wrap(consolidated), _wrap(recombined), _wrap(sanitized)]
        ),
    )
    mocker.patch.object(
        orchestrator_mod,
        "apply_operations",
        AsyncMock(return_value={"session_id": "s1"}),
    )

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error is None
    timeouts = [call.kwargs["timeout_seconds"] for call in llm_mock.call_args_list]
    assert timeouts == [
        orchestrator_mod.CONSOLIDATE_TIMEOUT_SECONDS,
        orchestrator_mod.RECOMBINE_TIMEOUT_SECONDS,
        orchestrator_mod.SANITIZE_TIMEOUT_SECONDS,
    ]


def test_long_output_phase_timeouts_exceed_the_shared_request_default():
    """Regression pin: every phase used to run on the shared 120s
    ``DEFAULT_REQUEST_TIMEOUT_SECONDS``, which cannot decode the 16384
    output tokens recombine/sanitize are budgeted for. If these ever
    drop back to (or below) the default, the token-cap raise becomes
    dead letter again."""
    assert orchestrator_mod.RECOMBINE_TIMEOUT_SECONDS > DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert orchestrator_mod.SANITIZE_TIMEOUT_SECONDS > DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert (
        orchestrator_mod.CONSOLIDATE_TIMEOUT_SECONDS >= DEFAULT_REQUEST_TIMEOUT_SECONDS
    )


def test_phase_timeouts_plus_headroom_fit_scheduler_and_lock_envelope():
    """Budget-math invariant with the drain budgeted as its OWN line item.

    The scheduler abandons the whole pass at
    ``SCHEDULER_DREAM_OPERATION_TIMEOUT_SECONDS`` (1800s), so the full pass
    — LLM phases + the ingestion drain + the remaining non-LLM headroom —
    must fit inside it with slack. The dream lock (``DEFAULT_LOCK_TTL_SECONDS``,
    1800s) is a separate, data-integrity boundary: apply RENEWS the lock to
    a fresh budget right before the drain (``LOCK_DRAIN_RENEWAL_SECONDS``),
    so the drain no longer counts against the original lock TTL — only the
    LLM phases + non-drain headroom do, and those must clear a 90%-of-TTL
    bar so a near-worst-case pass never reaches the lock expiry while writes
    are landing. A future bump to any phase timeout (or a cut to either
    envelope) fails here loudly."""
    llm = (
        orchestrator_mod.CONSOLIDATE_TIMEOUT_SECONDS
        + orchestrator_mod.RECOMBINE_TIMEOUT_SECONDS
        + orchestrator_mod.SANITIZE_TIMEOUT_SECONDS
    )
    non_drain_headroom = orchestrator_mod.DREAM_NON_LLM_HEADROOM_SECONDS
    drain = INGESTION_DRAIN_TIMEOUT_SECONDS

    # Whole pass (including the drain) must fit the scheduler abandonment
    # boundary with real margin.
    worst_case = llm + drain + non_drain_headroom
    assert worst_case < SCHEDULER_DREAM_OPERATION_TIMEOUT_SECONDS

    # The lock only has to cover everything UP TO the drain, because the
    # drain runs under a freshly renewed lock. Require 10% slack so the lock
    # never expires mid-write.
    assert llm + non_drain_headroom <= 0.9 * DEFAULT_LOCK_TTL_SECONDS

    # The renewal must comfortably outlast the drain cap it protects.
    assert LOCK_DRAIN_RENEWAL_SECONDS > drain


@pytest.mark.asyncio
async def test_consolidate_llm_failure_surfaces_error_and_skips_apply(mocker):
    """A failure in the consolidation step must surface as
    ``error="consolidate: ..."`` and never trigger apply_operations."""
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_build_input()),
    )
    mocker.patch.object(
        orchestrator_mod,
        "structured_completion",
        AsyncMock(side_effect=DreamLLMError("boom")),
    )
    apply_mock = mocker.patch.object(orchestrator_mod, "apply_operations", AsyncMock())

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error is not None
    assert result.error.startswith("consolidate:")
    apply_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_clamps_oversized_sanitizer_output(mocker):
    """The sanitizer model can over-emit; orchestrator must enforce caps."""
    # 1000 active facts -> 5% ceiling (50) sits above MAX_DEMOTIONS_PER_PASS,
    # so the absolute cap is the binding one for this assertion.
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_build_input(facts=1000)),
    )
    consolidated = ConsolidationOutput(facts=[])
    recombined = RecombinationOutput(proposals=[])

    # Build a sanitizer output that blows past every cap. Demotions
    # target real known fact uuids (f0..f99) so the clamp's known-uuid
    # pre-filter keeps them all and the cap is what binds.
    huge_sanitized = DreamOperations(
        writes=[ConsolidatedFact(content=f"w{i}", confidence=0.5) for i in range(100)],
        proposals=[
            ProposedFinding(
                content=f"p{i}",
                confidence=0.5,
                rationale="r",
            )
            for i in range(100)
        ],
        demotions=[DreamDemotion(edge_uuid=f"f{i}", reason="r") for i in range(100)],
        summary_for_user="ok",
    )

    mocker.patch.object(
        orchestrator_mod,
        "structured_completion",
        AsyncMock(
            side_effect=[_wrap(consolidated), _wrap(recombined), _wrap(huge_sanitized)]
        ),
    )

    captured: dict[str, DreamOperations] = {}

    async def fake_apply(
        user_id, pass_id, ops, *, known_fact_uuids=None, lock_handle=None
    ):
        captured["ops"] = ops
        return {
            "session_id": "s",
            "consolidated_count": len(ops.writes),
            "proposal_count": len(ops.proposals),
            "demotion_count": len(ops.demotions),
            "demotion_failed_count": 0,
            "entity_invalidation_count": 0,
        }

    mocker.patch.object(orchestrator_mod, "apply_operations", fake_apply)

    await orchestrator_mod.execute_dream_pass("u")

    assert captured["ops"] is not None
    from .prompts import (
        MAX_DEMOTIONS_PER_PASS,
        MAX_PROPOSALS_PER_PASS,
        MAX_WRITES_PER_PASS,
    )

    assert len(captured["ops"].writes) == MAX_WRITES_PER_PASS
    assert len(captured["ops"].proposals) == MAX_PROPOSALS_PER_PASS
    assert len(captured["ops"].demotions) == MAX_DEMOTIONS_PER_PASS


@pytest.mark.asyncio
async def test_demotions_capped_at_five_percent_of_active_facts(mocker):
    """A small active-fact set caps demotions below the absolute limit so
    one pass can't wipe a meaningful fraction of memory: 5% of 100 = 5,
    well under MAX_DEMOTIONS_PER_PASS (10)."""
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_build_input(facts=100)),
    )
    mocker.patch.object(
        orchestrator_mod,
        "structured_completion",
        AsyncMock(
            side_effect=[
                _wrap(ConsolidationOutput(facts=[])),
                _wrap(RecombinationOutput(proposals=[])),
                _wrap(
                    DreamOperations(
                        # Known fact uuids (f0..f49) so the clamp's
                        # pre-filter keeps them and the 5% cap binds.
                        demotions=[
                            DreamDemotion(edge_uuid=f"f{i}", reason="r")
                            for i in range(50)
                        ],
                        summary_for_user="ok",
                    )
                ),
            ]
        ),
    )

    captured: dict[str, DreamOperations] = {}

    async def fake_apply(
        user_id, pass_id, ops, *, known_fact_uuids=None, lock_handle=None
    ):
        captured["ops"] = ops
        return {
            "session_id": "s",
            "consolidated_count": 0,
            "proposal_count": 0,
            "demotion_count": len(ops.demotions),
            "demotion_failed_count": 0,
            "entity_invalidation_count": 0,
        }

    mocker.patch.object(orchestrator_mod, "apply_operations", fake_apply)

    await orchestrator_mod.execute_dream_pass("u")

    assert len(captured["ops"].demotions) == 5


def test_clamp_operations_demotion_cap_rules():
    """Unit-level coverage of the demotion ceiling: min(absolute, 5%), with
    an unknown count (-1) falling back to the absolute cap."""
    from .prompts import MAX_DEMOTIONS_PER_PASS

    ops = DreamOperations(
        demotions=[DreamDemotion(edge_uuid=f"e{i}", reason="r") for i in range(50)],
    )
    # 5% of 100 = 5 (below the absolute cap)
    assert len(orchestrator_mod._clamp_operations(ops, 100).demotions) == 5
    # 5% of 1000 = 50, so the absolute cap binds
    assert (
        len(orchestrator_mod._clamp_operations(ops, 1000).demotions)
        == MAX_DEMOTIONS_PER_PASS
    )
    # Unknown active-fact count -> absolute cap only, never zero
    assert (
        len(orchestrator_mod._clamp_operations(ops, -1).demotions)
        == MAX_DEMOTIONS_PER_PASS
    )


def test_clamp_operations_small_graph_demotion_cap_floors_at_one():
    """A small graph (< 20 active facts, where 5% rounds to 0) still gets
    a demotion budget of 1 — early-stage users must be able to demote a
    contradicted fact. Zero active facts means zero demotion budget:
    there is nothing legitimate to demote."""
    ops = DreamOperations(
        demotions=[DreamDemotion(edge_uuid=f"e{i}", reason="r") for i in range(50)],
    )
    assert len(orchestrator_mod._clamp_operations(ops, 10).demotions) == 1
    assert len(orchestrator_mod._clamp_operations(ops, 1).demotions) == 1
    assert len(orchestrator_mod._clamp_operations(ops, 19).demotions) == 1
    # 20 facts crosses the 5% threshold back to the proportional cap
    assert len(orchestrator_mod._clamp_operations(ops, 20).demotions) == 1
    assert len(orchestrator_mod._clamp_operations(ops, 40).demotions) == 2
    # No active facts at all -> no demotion budget
    assert len(orchestrator_mod._clamp_operations(ops, 0).demotions) == 0


def test_hallucinated_uuid_does_not_consume_cap_slot():
    """On a small graph the demotion cap floors at 1 — a hallucinated
    edge uuid at the head of the model's list must not eat that single
    slot and displace the valid demotion behind it. The clamp filters
    against known_fact_uuids BEFORE slicing to the cap."""
    ops = DreamOperations(
        demotions=[
            DreamDemotion(edge_uuid="hallucinated", reason="r"),
            DreamDemotion(edge_uuid="f0", reason="r"),
        ],
    )
    clamped = orchestrator_mod._clamp_operations(ops, 10, known_fact_uuids={"f0"})
    assert [d.edge_uuid for d in clamped.demotions] == ["f0"]

    # Without the allowlist the clamp can't pre-filter — the cap slices
    # the raw list and apply.py's filter remains the only defense.
    unfiltered = orchestrator_mod._clamp_operations(ops, 10)
    assert [d.edge_uuid for d in unfiltered.demotions] == ["hallucinated"]


@pytest.mark.asyncio
async def test_sync_path_filters_hallucinated_demotion_before_cap(mocker):
    """End-to-end on the sync path: 10 active facts → demotion cap 1; the
    sanitizer leads with a hallucinated uuid but the valid demotion (f0)
    is the one that survives clamping and reaches apply."""
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_build_input(facts=10)),
    )
    mocker.patch.object(
        orchestrator_mod,
        "structured_completion",
        AsyncMock(
            side_effect=[
                _wrap(ConsolidationOutput(facts=[])),
                _wrap(RecombinationOutput(proposals=[])),
                _wrap(
                    DreamOperations(
                        demotions=[
                            DreamDemotion(edge_uuid="hallucinated", reason="r"),
                            DreamDemotion(edge_uuid="f0", reason="r"),
                        ],
                        summary_for_user="ok",
                    )
                ),
            ]
        ),
    )

    captured: dict[str, DreamOperations] = {}

    async def fake_apply(
        user_id, pass_id, ops, *, known_fact_uuids=None, lock_handle=None
    ):
        captured["ops"] = ops
        return {
            "session_id": "s",
            "consolidated_count": 0,
            "proposal_count": 0,
            "demotion_count": len(ops.demotions),
            "demotion_failed_count": 0,
            "entity_invalidation_count": 0,
        }

    mocker.patch.object(orchestrator_mod, "apply_operations", fake_apply)

    await orchestrator_mod.execute_dream_pass("u")

    assert [d.edge_uuid for d in captured["ops"].demotions] == ["f0"]


def test_clamp_operations_caps_entity_invalidations():
    """Entity invalidations are the highest-blast-radius op (each one
    demotes every edge on the entity), so the clamp must bound their
    count at MAX_ENTITY_INVALIDATIONS_PER_PASS — they used to pass
    through entirely uncapped."""
    ops = DreamOperations(
        entity_invalidations=[
            EntityInvalidation(entity_uuid=f"ent{i}", reason="r") for i in range(25)
        ],
    )
    clamped = orchestrator_mod._clamp_operations(ops, 100)
    assert (
        len(clamped.entity_invalidations)
        == orchestrator_mod.MAX_ENTITY_INVALIDATIONS_PER_PASS
    )
    # The first N proposed invalidations survive, in order
    assert [e.entity_uuid for e in clamped.entity_invalidations] == [
        "ent0",
        "ent1",
    ]


@pytest.mark.asyncio
async def test_sync_path_passes_known_fact_uuids_to_apply(mocker):
    """The sync orchestrator must thread the input bundle's
    known_fact_uuids into apply_operations so the demotion pre-flight
    filter (apply.py) can reject hallucinated edge uuids."""
    input_bundle = _build_input(facts=3)
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=input_bundle),
    )
    mocker.patch.object(
        orchestrator_mod,
        "structured_completion",
        AsyncMock(
            side_effect=[
                _wrap(ConsolidationOutput(facts=[])),
                _wrap(RecombinationOutput(proposals=[])),
                _wrap(DreamOperations(summary_for_user="ok")),
            ]
        ),
    )
    apply_mock = mocker.patch.object(
        orchestrator_mod,
        "apply_operations",
        AsyncMock(
            return_value={
                "session_id": "s",
                "consolidated_count": 0,
                "proposal_count": 0,
                "demotion_count": 0,
                "demotion_failed_count": 0,
                "entity_invalidation_count": 0,
            }
        ),
    )

    await orchestrator_mod.execute_dream_pass("u")

    apply_mock.assert_awaited_once()
    assert (
        apply_mock.await_args.kwargs["known_fact_uuids"]
        == input_bundle.known_fact_uuids
    )


@pytest.mark.asyncio
async def test_lock_held_returns_skipped_lock_held(mocker):
    from .locks import DreamLockHeld

    @asynccontextmanager
    async def busy_lock(*args, **kwargs):
        raise DreamLockHeld(args[0] if args else "?")
        yield  # pragma: no cover

    mocker.patch.object(orchestrator_mod, "dream_lock", busy_lock)
    fetch_mock = mocker.patch.object(
        orchestrator_mod, "gather_dream_input", AsyncMock()
    )

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.skipped is True
    assert result.skip_reason == "lock_held"
    fetch_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_budget_skip_returns_insufficient_credits_without_running_phases(mocker):
    """Pre-flight rate-limit cap exceeded → skipped, no LLM calls, no apply."""
    mocker.patch.object(
        orchestrator_mod,
        "check_dream_budget",
        AsyncMock(return_value=(False, "insufficient_credits")),
    )
    structured = mocker.patch.object(
        orchestrator_mod, "structured_completion", AsyncMock()
    )
    apply_mock = mocker.patch.object(orchestrator_mod, "apply_operations", AsyncMock())
    fetch_mock = mocker.patch.object(
        orchestrator_mod, "gather_dream_input", AsyncMock()
    )

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.skipped is True
    assert result.skip_reason == "insufficient_credits"
    structured.assert_not_called()
    apply_mock.assert_not_awaited()
    fetch_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_budget_check_failing_closed_surfaces_as_error(mocker):
    """Redis brown-out during pre-flight → error, NOT skipped, so the
    admin endpoint surfaces it and the scheduler retries next tick."""
    mocker.patch.object(
        orchestrator_mod,
        "check_dream_budget",
        AsyncMock(return_value=(False, "rate_limit_unavailable")),
    )
    apply_mock = mocker.patch.object(orchestrator_mod, "apply_operations", AsyncMock())

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error is not None
    assert "rate_limit_unavailable" in result.error
    assert result.skipped is False
    apply_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_each_completed_phase_charges_once(mocker):
    """One billing row per LLM call — verifies the chat convention
    (per-call rows) is preserved end-to-end in the orchestrator."""
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_build_input()),
    )
    consolidated = ConsolidationOutput(facts=[])
    recombined = RecombinationOutput(proposals=[])
    sanitized = DreamOperations()
    mocker.patch.object(
        orchestrator_mod,
        "structured_completion",
        AsyncMock(
            side_effect=[_wrap(consolidated), _wrap(recombined), _wrap(sanitized)]
        ),
    )
    mocker.patch.object(
        orchestrator_mod,
        "apply_operations",
        AsyncMock(
            return_value={
                "session_id": "s",
                "consolidated_count": 0,
                "proposal_count": 0,
                "demotion_count": 0,
                "demotion_failed_count": 0,
                "entity_invalidation_count": 0,
            }
        ),
    )
    charge_spy = AsyncMock()
    mocker.patch.object(orchestrator_mod, "record_phase_cost", charge_spy)

    await orchestrator_mod.execute_dream_pass("u")

    assert charge_spy.await_count == 3
    phases_charged = [c.kwargs["phase_usage"].phase for c in charge_spy.await_args_list]
    assert phases_charged == ["consolidate", "recombine", "sanitize"]


@pytest.mark.asyncio
async def test_partial_failure_still_charges_completed_phases(mocker):
    """If recombine errors after consolidate succeeded, we must still
    bill for the consolidate tokens — we already paid the provider."""
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_build_input()),
    )
    consolidated = ConsolidationOutput(facts=[])
    mocker.patch.object(
        orchestrator_mod,
        "structured_completion",
        AsyncMock(side_effect=[_wrap(consolidated), DreamLLMError("recombine boom")]),
    )
    apply_mock = mocker.patch.object(orchestrator_mod, "apply_operations", AsyncMock())
    charge_spy = AsyncMock()
    mocker.patch.object(orchestrator_mod, "record_phase_cost", charge_spy)

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error is not None
    assert result.error.startswith("recombine:")
    assert charge_spy.await_count == 1  # consolidate charged, recombine never ran
    assert charge_spy.await_args.kwargs["phase_usage"].phase == "consolidate"
    apply_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# No-new-activity skip — the per-user ``dream:last_completed:{user_id}``
# marker short-circuits the pass before phase 1 when nothing new landed.
# ---------------------------------------------------------------------------


def _input_with_episodes(episodes: list[EpisodeRow]) -> DreamInput:
    """Bundle with one old active fact + the given episodes — the 'old
    facts, maybe-stale episodes' nightly shape."""
    return DreamInput(
        user_id="u",
        group_id="g",
        window_start=datetime(2026, 5, 26, tzinfo=timezone.utc),
        window_end=datetime(2026, 6, 9, tzinfo=timezone.utc),
        episodes=episodes,
        facts=[
            FactRow(
                uuid="f0",
                source="A",
                target="B",
                name="likes",
                fact="A likes B",
                scope="real:global",
                confidence=0.7,
                status="active",
                created_at="2026-01-01T00:00:00Z",
            )
        ],
        recent_sessions=[],
        known_fact_uuids={"f0"},
        known_episode_uuids={e.uuid for e in episodes},
    )


def _episode_at(
    valid_at: str | None,
    *,
    uuid: str = "e0",
    name: str | None = None,
    source_description: str | None = None,
    created_at: str | None = None,
) -> EpisodeRow:
    return EpisodeRow(
        uuid=uuid,
        name=name,
        content="hello",
        source_description=source_description,
        valid_at=valid_at,
        created_at=created_at,
    )


def _input_with_episode_times(*valid_ats: str | None) -> DreamInput:
    """Bundle with plain user episodes at the given ``valid_at`` timestamps."""
    return _input_with_episodes(
        [_episode_at(valid_at, uuid=f"e{i}") for i, valid_at in enumerate(valid_ats)]
    )


def _stub_three_phases_and_apply(mocker) -> AsyncMock:
    """Minimal happy-path pipeline: empty phase outputs + a stubbed apply.
    Returns the apply mock so callers can assert the pass actually ran."""
    mocker.patch.object(
        orchestrator_mod,
        "structured_completion",
        AsyncMock(
            side_effect=[
                _wrap(ConsolidationOutput(facts=[])),
                _wrap(RecombinationOutput(proposals=[])),
                _wrap(DreamOperations(summary_for_user="ok")),
            ]
        ),
    )
    return mocker.patch.object(
        orchestrator_mod,
        "apply_operations",
        AsyncMock(
            return_value={
                "session_id": "s",
                "consolidated_count": 0,
                "proposal_count": 0,
                "demotion_count": 0,
                "demotion_failed_count": 0,
                "entity_invalidation_count": 0,
            }
        ),
    )


@pytest.mark.asyncio
async def test_marker_newer_than_all_episodes_skips_with_no_new_activity(
    mocker, _stub_marker_redis
):
    """Every episode predates the last completed pass ⇒ skip before
    phase 1: no LLM calls, no apply, no fresh marker stamp."""
    _stub_marker_redis.get.return_value = "2026-06-08T00:00:00+00:00"
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(
            return_value=_input_with_episode_times(
                "2026-06-01T00:00:00Z", "2026-06-07T23:59:59Z"
            )
        ),
    )
    structured = mocker.patch.object(
        orchestrator_mod, "structured_completion", AsyncMock()
    )
    apply_mock = mocker.patch.object(orchestrator_mod, "apply_operations", AsyncMock())

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.skipped is True
    assert result.skip_reason == "no_new_activity"
    structured.assert_not_called()
    apply_mock.assert_not_awaited()
    _stub_marker_redis.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_marker_with_old_facts_and_no_episodes_skips(mocker, _stub_marker_redis):
    """The literal nightly bug shape: old active facts, zero episodes in
    the window, marker present ⇒ no_new_activity (NOT no_input — facts
    exist, so the old code would have run all three phases)."""
    _stub_marker_redis.get.return_value = "2026-06-08T00:00:00+00:00"
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_input_with_episode_times()),
    )
    structured = mocker.patch.object(
        orchestrator_mod, "structured_completion", AsyncMock()
    )

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.skipped is True
    assert result.skip_reason == "no_new_activity"
    structured.assert_not_called()


@pytest.mark.asyncio
async def test_no_marker_runs_full_pass(mocker, _stub_marker_redis):
    """First-ever pass (or expired marker): get → None means there's no
    baseline to compare against, so the pass runs."""
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_input_with_episode_times("2026-06-01T00:00:00Z")),
    )
    apply_mock = _stub_three_phases_and_apply(mocker)

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.skipped is False
    assert result.error is None
    apply_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_redis_error_on_marker_read_fails_open_and_runs(
    mocker, _stub_marker_redis
):
    """A Redis blip on the marker read must never block a dream — the
    marker only exists to save LLM spend."""
    _stub_marker_redis.get.side_effect = ConnectionError("redis down")
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_input_with_episode_times("2026-06-01T00:00:00Z")),
    )
    apply_mock = _stub_three_phases_and_apply(mocker)

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.skipped is False
    assert result.error is None
    apply_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_corrupt_marker_bytes_fail_open_and_run(mocker, _stub_marker_redis):
    """A marker holding invalid UTF-8 bytes must fail open like any other
    unparseable value — never escalate the cost optimization into a pass
    failure."""
    _stub_marker_redis.get.return_value = b"\xff\xfe corrupt"
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_input_with_episode_times("2026-06-01T00:00:00Z")),
    )
    apply_mock = _stub_three_phases_and_apply(mocker)

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.skipped is False
    assert result.error is None
    apply_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_episode_newer_than_marker_runs_pass(mocker, _stub_marker_redis):
    """One episode landed after the last completed pass ⇒ there's new
    material to consolidate, so the pass runs."""
    _stub_marker_redis.get.return_value = "2026-06-05T00:00:00+00:00"
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(
            return_value=_input_with_episode_times(
                "2026-06-01T00:00:00Z", "2026-06-08T12:00:00Z"
            )
        ),
    )
    apply_mock = _stub_three_phases_and_apply(mocker)

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.skipped is False
    apply_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_unparseable_episode_timestamp_counts_as_new_and_runs(
    mocker, _stub_marker_redis
):
    """An episode whose valid_at/created_at can't be parsed can't be
    proven old — fail open and run the pass."""
    _stub_marker_redis.get.return_value = "2026-06-08T00:00:00+00:00"
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_input_with_episode_times(None)),
    )
    apply_mock = _stub_three_phases_and_apply(mocker)

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.skipped is False
    apply_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_dream_authored_episode_newer_than_marker_does_not_count_as_new(
    mocker, _stub_marker_redis
):
    """A productive pass enqueues its own ``dream_``-prefixed episodes with
    ``valid_at = now()`` (after the stamped ``window_end``). They must NOT
    count as new activity, or every productive night would trigger a paid
    no-op pass the next night that only re-reads its own output."""
    _stub_marker_redis.get.return_value = "2026-06-08T00:00:00+00:00"
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(
            return_value=_input_with_episodes(
                [
                    _episode_at("2026-06-01T00:00:00Z", uuid="e0"),
                    _episode_at(
                        "2026-06-30T00:00:00Z",
                        uuid="e1",
                        name="dream_p-xyz_consolidate_000",
                        source_description="dream-pass consolidation; src_episodes=e0",
                    ),
                ]
            )
        ),
    )
    structured = mocker.patch.object(
        orchestrator_mod, "structured_completion", AsyncMock()
    )
    apply_mock = mocker.patch.object(orchestrator_mod, "apply_operations", AsyncMock())

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.skipped is True
    assert result.skip_reason == "no_new_activity"
    structured.assert_not_called()
    apply_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_normal_episode_newer_than_marker_runs_even_beside_dream_episode(
    mocker, _stub_marker_redis
):
    """Excluding dream-authored episodes must not suppress genuine user
    activity: a normal episode newer than the marker still triggers the pass
    even when a dream-authored episode sits alongside it."""
    _stub_marker_redis.get.return_value = "2026-06-08T00:00:00+00:00"
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(
            return_value=_input_with_episodes(
                [
                    _episode_at(
                        "2026-06-30T00:00:00Z",
                        uuid="e0",
                        name="dream_p-xyz_recombine_000",
                        source_description="dream-pass proposal",
                    ),
                    _episode_at("2026-06-09T12:00:00Z", uuid="e1"),
                ]
            )
        ),
    )
    apply_mock = _stub_three_phases_and_apply(mocker)

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.skipped is False
    apply_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_episode_created_after_marker_counts_as_new_despite_old_valid_at(
    mocker, _stub_marker_redis
):
    """An episode whose graph node is materialized by async ingestion after
    the gather query has ``valid_at`` (reference_time) < marker but
    ``created_at`` > marker. Novelty uses the later of the two, so it counts
    as new instead of being silently dropped."""
    _stub_marker_redis.get.return_value = "2026-06-08T00:00:00+00:00"
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(
            return_value=_input_with_episodes(
                [
                    _episode_at(
                        "2026-06-01T00:00:00Z",
                        uuid="e0",
                        created_at="2026-06-09T00:00:00Z",
                    )
                ]
            )
        ),
    )
    apply_mock = _stub_three_phases_and_apply(mocker)

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.skipped is False
    apply_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_episode_old_in_both_valid_at_and_created_at_skips(
    mocker, _stub_marker_redis
):
    """Taking the max of valid_at/created_at must not spuriously run: when
    both timestamps predate the marker, the pass still skips."""
    _stub_marker_redis.get.return_value = "2026-06-08T00:00:00+00:00"
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(
            return_value=_input_with_episodes(
                [
                    _episode_at(
                        "2026-06-01T00:00:00Z",
                        uuid="e0",
                        created_at="2026-06-05T00:00:00Z",
                    )
                ]
            )
        ),
    )
    structured = mocker.patch.object(
        orchestrator_mod, "structured_completion", AsyncMock()
    )
    apply_mock = mocker.patch.object(orchestrator_mod, "apply_operations", AsyncMock())

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.skipped is True
    assert result.skip_reason == "no_new_activity"
    structured.assert_not_called()
    apply_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_pass_stamps_last_completed_marker(mocker, _stub_marker_redis):
    """A non-skipped sync apply stamps ``dream:last_completed:{user_id}``
    with an ISO timestamp and the ~35-day TTL so the next nightly pass
    can skip when nothing new landed."""
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_input_with_episode_times("2026-06-08T12:00:00Z")),
    )
    _stub_three_phases_and_apply(mocker)

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.skipped is False
    assert result.error is None
    _stub_marker_redis.set.assert_awaited_once()
    set_args = _stub_marker_redis.set.await_args
    assert set_args.args[0] == "dream:last_completed:u"
    # Stamped with the gather-window end (NOT apply-completion time) so
    # episodes that arrive mid-pass still count as new next time.
    stamped = datetime.fromisoformat(set_args.args[1])
    assert stamped == datetime(2026, 6, 9, tzinfo=timezone.utc)
    assert set_args.kwargs == {"ex": orchestrator_mod.LAST_COMPLETED_TTL_SECONDS}


@pytest.mark.asyncio
async def test_admin_trigger_bypasses_no_new_activity_marker(
    mocker, _stub_marker_redis
):
    """status_id is set only by the admin 'dream now' trigger — the
    memory-debugging path must run the full pipeline even when the marker
    is newer than every episode, otherwise a re-run after prompt/flag
    changes silently no-ops for up to the marker TTL."""
    _stub_marker_redis.get.return_value = "2026-06-08T00:00:00+00:00"
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_input_with_episode_times("2026-06-01T00:00:00Z")),
    )
    apply_mock = _stub_three_phases_and_apply(mocker)

    result = await orchestrator_mod.execute_dream_pass("u", status_id="job-1")

    assert result.skipped is False
    assert result.error is None
    apply_mock.assert_awaited_once()
    # The bypass must not even read the marker — the skip is caller-gated.
    _stub_marker_redis.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_pass_does_not_stamp_marker(mocker, _stub_marker_redis):
    """An LLM failure means no apply ran — the marker must NOT advance,
    otherwise tomorrow's pass would skip material this one never chewed."""
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_input_with_episode_times("2026-06-08T12:00:00Z")),
    )
    mocker.patch.object(
        orchestrator_mod,
        "structured_completion",
        AsyncMock(side_effect=DreamLLMError("boom")),
    )

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error is not None
    _stub_marker_redis.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_apply_stats_leave_dream_session_id_none(mocker):
    """When apply skips the dream session (empty pass), its stats carry
    no session_id — the result must surface ``dream_session_id=None``,
    not an empty string."""
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_build_input()),
    )
    mocker.patch.object(
        orchestrator_mod,
        "structured_completion",
        AsyncMock(
            side_effect=[
                _wrap(ConsolidationOutput(facts=[])),
                _wrap(RecombinationOutput(proposals=[])),
                _wrap(DreamOperations(summary_for_user="Nothing new tonight.")),
            ]
        ),
    )
    mocker.patch.object(
        orchestrator_mod,
        "apply_operations",
        AsyncMock(
            return_value={
                "consolidated_count": 0,
                "proposal_count": 0,
                "demotion_count": 0,
                "demotion_failed_count": 0,
                "entity_invalidation_count": 0,
            }
        ),
    )

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error is None
    assert result.dream_session_id is None
    assert result.summary_for_user == "Nothing new tonight."


_ = MagicMock  # keep import for editor convenience; not directly used


class TestTransientIntentFilter:
    """#13388: transient-intent 'facts' (questions captured as memory) are
    dropped deterministically; durable facts and goals are kept; generic
    world-knowledge is intentionally NOT handled here (prompt's job)."""

    @pytest.mark.parametrize(
        "content",
        [
            "User is asking how Kubernetes works",
            # 'asking' is transient only for interrogative questions, not
            # durable 'asking FOR/TO' requests (#3622917532).
            "User is asking how to deploy",
            "User is interested in knowing which Pull Requests they currently have open",
            "The user is wondering about the deploy timeline",
            "user wants to know the markup on credits",
            "User asked whether the migration is risky",
            "User is curious about lighthouse restorations",
            "User is trying to understand the auth flow",
            # 'learn'/'understand'/'find out'/'interested in learning' ARE
            # transient when followed by an interrogative (a question, not a
            # durable skill goal / aspiration).
            "User wants to learn how Kubernetes networking works",
            "User is interested in learning what the credit markup is",
            "User wants to understand how the auth flow works",
            "User wants to find out what the credit markup is",
            # 'confused about' / 'unsure about' require the 'about' complement
            # and drop the transient question form (#3623648910 / #3623648918).
            "User is confused about the auth flow",
            "User is unsure about the migration risk",
            # 'asking' interrogatives beyond 'how' — 'which' / 'about' branches
            # must fire too (#3623648935).
            "User is asking which regions are cheapest",
            "User asked about the credit markup",
            # Perfect-progressive auxiliary ('has been asking') must be caught
            # by the widened leading-auxiliary group (#3623648975).
            "User has been asking how the auth flow works",
        ],
    )
    def test_flags_transient_intent(self, content):
        assert orchestrator_mod._is_transient_intent(content) is True

    @pytest.mark.parametrize(
        "content",
        [
            # Durable goals — NOT transient, must be kept.
            "User wants to create Bluesky blocks",
            "User is migrating auth to Better Auth",
            "User is building a leaf-blower-simulator game",
            # Durable SKILL goals / aspirations — 'learn/understand/find out
            # <thing>' with no interrogative is an aspiration, not a question
            # (review FP fixes #3622917532 / #3622917546).
            "User wants to learn Spanish",
            "User is interested in learning Rust as their next language",
            "User wants to understand distributed systems",
            "User wants to find out about new markets to expand into",
            # Durable REQUESTS — 'asking for/to' is semantically 'wants X',
            # not a knowledge-seeking question (#3622917532).
            "User is asking for weekly reports",
            "User is asking the agent to monitor open PRs",
            # Durable personality trait — 'curious by nature' needs no
            # 'about' complement, so it must survive (#3622917543).
            "User is curious by nature",
            # 'confused'/'unsure' without the 'about' complement are durable
            # traits, not questions — keep-pins the complement guard so a
            # future broadening fails loudly (#3623648910).
            "User is confused by nature",
            # Name-subject scope guard: the deterministic gate only covers the
            # generic 'user' subject; name-phrased transient intent is left to
            # the sanitize prompt, so it must NOT be dropped here
            # (#3623649071 / #3623649079).
            "Nick is asking how to scale the cluster",
            # Bare 'interested in <noun>' must never match — pins the carve-out
            # so a future broadening of the regex fails loudly here.
            "User is interested in lighthouses",
            "User is interested in knowledge graphs",
            # Durable facts about the user.
            "Nick's favorite deployment region is us-east1",
            "Nick prefers Python for backend services",
            # Generic world-knowledge — left to the sanitize PROMPT, the
            # code filter must NOT drop these (would risk false positives).
            "Kubernetes uses pods as the smallest unit",
            "Better Auth supports dual JWT validation",
        ],
    )
    def test_keeps_durable_facts_goals_and_generic_knowledge(self, content):
        assert orchestrator_mod._is_transient_intent(content) is False

    def test_clamp_drops_transient_writes_and_proposals(self):
        ops = DreamOperations(
            writes=[
                ConsolidatedFact(
                    content="User is asking how K8s works", confidence=0.5
                ),
                ConsolidatedFact(content="Nick prefers Python", confidence=0.9),
            ],
            proposals=[
                ProposedFinding(
                    content="User wants to know the credit markup",
                    confidence=0.5,
                    rationale="x",
                    source_fact_uuids=["f1"],
                ),
                ProposedFinding(
                    content="Nick and Sarah both work on auth",
                    confidence=0.6,
                    rationale="y",
                    source_fact_uuids=["f2"],
                ),
            ],
            summary_for_user="ok",
        )
        clamped = orchestrator_mod._clamp_operations(ops, active_fact_count=50)
        assert [w.content for w in clamped.writes] == ["Nick prefers Python"]
        assert [p.content for p in clamped.proposals] == [
            "Nick and Sarah both work on auth"
        ]

    def test_drop_transient_intent_returns_kept_and_count(self):
        """Directly pins both outputs of _drop_transient_intent so an
        off-by-one in the dropped_count feeding logger.info is caught."""
        items = [
            ConsolidatedFact(content="User is asking how K8s works", confidence=0.5),
            ConsolidatedFact(content="Nick prefers Python", confidence=0.9),
            ConsolidatedFact(
                content="User wants to know the credit markup", confidence=0.4
            ),
            ConsolidatedFact(content="Nick deploys to us-east1", confidence=0.8),
        ]
        kept, dropped_count = orchestrator_mod._drop_transient_intent(items)
        assert [k.content for k in kept] == [
            "Nick prefers Python",
            "Nick deploys to us-east1",
        ]
        assert dropped_count == 2

    def test_drop_transient_intent_empty_and_none_dropped(self):
        """No transient items → all kept, zero dropped."""
        items = [
            ConsolidatedFact(content="Nick prefers Python", confidence=0.9),
        ]
        kept, dropped_count = orchestrator_mod._drop_transient_intent(items)
        assert [k.content for k in kept] == ["Nick prefers Python"]
        assert dropped_count == 0


class TestNearDuplicateWriteDedup:
    """#13387: a single pass that emits the same fact phrased multiple ways
    is collapsed to the longest (canonical) phrasing; genuinely distinct
    facts — even about the same entity — are preserved."""

    def test_collapses_near_identical_phrasings_keeping_longest(self):
        writes = [
            ConsolidatedFact(
                content="Nick uses Terminus on his iPhone for CLI work",
                confidence=0.6,
            ),
            ConsolidatedFact(
                content=(
                    "Nick uses Terminus on his iPhone for CLI work and wants "
                    "it to display more ASCII characters"
                ),
                confidence=0.7,
            ),
        ]
        kept, dropped = orchestrator_mod._dedupe_near_duplicate_writes(writes)
        assert dropped == 1
        assert len(kept) == 1
        # The longer, more specific phrasing survives.
        assert "more ASCII characters" in kept[0].content

    def test_keeps_distinct_facts_about_same_entity(self):
        writes = [
            ConsolidatedFact(content="Nick prefers Python for backend", confidence=0.8),
            ConsolidatedFact(
                content="Nick prefers Rust for systems work", confidence=0.8
            ),
        ]
        kept, dropped = orchestrator_mod._dedupe_near_duplicate_writes(writes)
        assert dropped == 0
        assert len(kept) == 2

    def test_one_distinguishing_word_is_not_a_duplicate(self):
        """Containment guard: two facts differing by a single key word
        (auth vs billing) must NOT be merged despite high overlap."""
        writes = [
            ConsolidatedFact(
                content="Nick deployed the auth service to prod", confidence=0.7
            ),
            ConsolidatedFact(
                content="Nick deployed the billing service to prod", confidence=0.7
            ),
        ]
        kept, dropped = orchestrator_mod._dedupe_near_duplicate_writes(writes)
        assert dropped == 0
        assert len(kept) == 2

    def test_preserves_original_order_of_survivors(self):
        writes = [
            ConsolidatedFact(content="Alpha fact about onboarding", confidence=0.5),
            ConsolidatedFact(content="Beta fact about billing", confidence=0.5),
            ConsolidatedFact(content="Gamma fact about deploys", confidence=0.5),
        ]
        kept, dropped = orchestrator_mod._dedupe_near_duplicate_writes(writes)
        assert dropped == 0
        assert [w.content for w in kept] == [
            "Alpha fact about onboarding",
            "Beta fact about billing",
            "Gamma fact about deploys",
        ]

    def test_identical_content_in_different_scopes_is_kept(self):
        writes = [
            ConsolidatedFact(
                content="Nick uses Terminus on his iPhone for CLI work",
                scope="real:global",
                confidence=0.7,
            ),
            ConsolidatedFact(
                content="Nick uses Terminus on his iPhone for CLI work",
                scope="project:foo",
                confidence=0.7,
            ),
        ]
        kept, dropped = orchestrator_mod._dedupe_near_duplicate_writes(writes)
        assert dropped == 0
        assert len(kept) == 2

    def test_survivor_absorbs_dropped_writes_provenance(self):
        writes = [
            ConsolidatedFact(
                content="Nick uses Terminus on his iPhone for CLI work",
                confidence=0.6,
                source_episode_uuids=["ep-1", "ep-2"],
            ),
            ConsolidatedFact(
                content=(
                    "Nick uses Terminus on his iPhone for CLI work and wants "
                    "it to display more ASCII characters"
                ),
                confidence=0.7,
                source_episode_uuids=["ep-2", "ep-3"],
            ),
        ]
        kept, dropped = orchestrator_mod._dedupe_near_duplicate_writes(writes)
        assert dropped == 1
        assert len(kept) == 1
        # Survivor keeps its own uuids first, then the absorbed extras.
        assert kept[0].source_episode_uuids == ["ep-2", "ep-3", "ep-1"]

    def test_word_order_permutation_is_not_merged(self):
        writes = [
            ConsolidatedFact(content="Alice introduced Bob to Carol", confidence=0.7),
            ConsolidatedFact(content="Alice introduced Carol to Bob", confidence=0.7),
        ]
        kept, dropped = orchestrator_mod._dedupe_near_duplicate_writes(writes)
        assert dropped == 0
        assert len(kept) == 2

    def test_negated_fact_is_not_merged(self):
        writes = [
            ConsolidatedFact(content="Nick uses vim", confidence=0.7),
            ConsolidatedFact(content="Nick never uses vim", confidence=0.7),
        ]
        kept, dropped = orchestrator_mod._dedupe_near_duplicate_writes(writes)
        assert dropped == 0
        assert len(kept) == 2

    def test_clamp_collapses_duplicate_writes(self):
        ops = DreamOperations(
            writes=[
                ConsolidatedFact(
                    content="Churn rate rose sharply in Q2", confidence=0.6
                ),
                ConsolidatedFact(
                    content="The churn rate rose sharply during Q2 this year",
                    confidence=0.7,
                ),
                ConsolidatedFact(
                    content="Revenue grew 12 percent in Q2", confidence=0.8
                ),
            ],
            proposals=[],
            summary_for_user="ok",
        )
        clamped = orchestrator_mod._clamp_operations(ops, active_fact_count=50)
        contents = [w.content for w in clamped.writes]
        # The two churn paraphrases collapse to one; revenue fact untouched.
        assert len(contents) == 2
        assert any("Revenue grew" in c for c in contents)
        assert sum("churn rate rose" in c.lower() for c in contents) == 1


@pytest.mark.asyncio
async def test_batch_handoff_revokes_batch_when_lock_extend_fails(mocker):
    """If the dream lock cannot be extended to the batch window, the
    just-submitted batch must never be applied: the pending entry is
    removed (poller never dispatches callbacks), the input bundle is
    dropped, the lock is NOT disowned, and the pass reports failure."""
    submission = mocker.MagicMock(provider_batch_id="batch-xyz")
    mocker.patch("backend.copilot.dream.batch_submit.persist_input_bundle", AsyncMock())
    mocker.patch(
        "backend.copilot.dream.batch_submit.phase_models_for_config",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "backend.copilot.dream.batch_submit.submit_phase",
        AsyncMock(return_value=submission),
    )
    remove_pending = mocker.patch(
        "backend.executor.batch_executor.remove_pending", AsyncMock()
    )
    delete_bundle = mocker.patch(
        "backend.copilot.dream.batch_submit.delete_input_bundle", AsyncMock()
    )
    handle = mocker.MagicMock()
    handle.extend = AsyncMock(return_value=False)
    handle.token = "tok"

    config = mocker.MagicMock()
    config.direct_anthropic_api_key = "key"

    result = await orchestrator_mod._submit_dream_pass_batch(
        user_id="u-lock-lost",
        pass_id="p-lock-lost",
        started_at=datetime.now(timezone.utc),
        monotonic_start=0.0,
        execution_path="anthropic_batch",
        config=config,
        input_bundle=_build_input(),
        status_id=None,
        dream_lock_handle=handle,
    )

    remove_pending.assert_awaited_once_with("batch-xyz")
    delete_bundle.assert_awaited_once_with("p-lock-lost")
    handle.disown.assert_not_called()
    assert result.error and "lock lost" in result.error
