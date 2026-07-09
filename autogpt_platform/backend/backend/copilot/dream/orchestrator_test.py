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

from . import orchestrator as orchestrator_mod
from .fetch import DreamInput, EpisodeRow, FactRow
from .llm import CompletionUsage, DreamLLMError, StructuredCompletion
from .schemas import (
    ConsolidatedFact,
    ConsolidationOutput,
    DreamDemotion,
    DreamOperations,
    EntityInvalidation,
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
    apply_mock.assert_awaited_once()


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

    async def fake_apply(user_id, pass_id, ops, *, known_fact_uuids=None):
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

    async def fake_apply(user_id, pass_id, ops, *, known_fact_uuids=None):
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

    async def fake_apply(user_id, pass_id, ops, *, known_fact_uuids=None):
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


_ = MagicMock  # keep import for editor convenience; not directly used
