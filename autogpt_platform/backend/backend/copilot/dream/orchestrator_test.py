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
    mocker.patch.object(
        orchestrator_mod,
        "gather_dream_input",
        AsyncMock(return_value=_build_input()),
    )
    consolidated = ConsolidationOutput(facts=[])
    recombined = RecombinationOutput(proposals=[])

    # Build a sanitizer output that blows past every cap.
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
        demotions=[DreamDemotion(edge_uuid=f"e{i}", reason="r") for i in range(100)],
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

    async def fake_apply(user_id, pass_id, ops):
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


_ = MagicMock  # keep import for editor convenience; not directly used
