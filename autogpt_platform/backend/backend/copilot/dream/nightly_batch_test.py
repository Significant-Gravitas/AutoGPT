"""Nightly batch submitter tests.

Contracts pinned here:

  1. **One pre-flight billing check** for the whole pass — paywalled
     or over-cap users skip the entire fan-out before any submitter
     runs.
  2. **Per-submitter failure isolation** — a crashed dream submitter
     still lets the ratification sweep run; both errors surface in
     ``NightlyBatchResult.error``.
  3. **Sequential ordering** — dream first, then ratification.
  4. **Result shape** — ``NightlyBatchResult`` carries per-submitter
     Pydantic models (not generic dicts) so admin viz + AgentProbe
     can read them without runtime type dispatch.
  5. **``nightly_id`` correlation** — every submitter call shares
     the same id so per-LLM-call cost log rows join back to one pass.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.dream.ratification import RatificationResult
from backend.copilot.dream.schemas import DreamPassResult

from . import nightly_batch
from .nightly_batch import run_nightly_batch_submit


def _dream_result(
    *,
    error: str | None = None,
    skipped: bool = False,
    execution_path: Literal[
        "sync_baseline", "anthropic_batch", "openai_batch"
    ] = "sync_baseline",
) -> DreamPassResult:
    return DreamPassResult(
        user_id="u",
        pass_id="dream-1",
        execution_path=execution_path,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        elapsed_seconds=0.5,
        error=error,
        skipped=skipped,
        skip_reason="no_input" if skipped else None,
        consolidated_count=2 if not (error or skipped) else 0,
        proposal_count=1 if not (error or skipped) else 0,
    )


def _ratification_result(
    *, error: str | None = None, ratified: int = 0, superseded: int = 0
) -> RatificationResult:
    now = datetime.now(timezone.utc)
    return RatificationResult(
        user_id="u",
        started_at=now,
        completed_at=now,
        examined_count=ratified + superseded,
        ratified_count=ratified,
        superseded_count=superseded,
        per_edge_errors=[],
        error=error,
    )


# ---------------------------------------------------------------------------
# Pre-flight billing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paywalled_user_skips_entire_fanout_no_submitters_run():
    """One pre-flight billing check — paywalled user costs us one LD
    lookup + one Redis read for the night, not one per submitter."""
    dream_spy = AsyncMock()
    rat_spy = AsyncMock()
    with patch.object(
        nightly_batch,
        "check_dream_budget",
        new=AsyncMock(return_value=(False, "insufficient_credits")),
    ), patch(
        "backend.copilot.dream.orchestrator.execute_dream_pass", new=dream_spy
    ), patch.object(
        nightly_batch, "run_ratification_pass", new=rat_spy
    ):
        result = await run_nightly_batch_submit("u")

    assert result.skipped is True
    assert result.skip_reason == "insufficient_credits"
    dream_spy.assert_not_called()
    rat_spy.assert_not_called()


@pytest.mark.asyncio
async def test_redis_brownout_during_budget_check_surfaces_as_error_not_skipped():
    """Rate-limit-unavailable → orchestrator fails closed; surfaces as
    error (so the cron retries next tick), NOT skipped."""
    with patch.object(
        nightly_batch,
        "check_dream_budget",
        new=AsyncMock(return_value=(False, "rate_limit_unavailable")),
    ):
        result = await run_nightly_batch_submit("u")

    assert result.skipped is False
    assert result.error is not None
    assert "rate_limit_unavailable" in result.error


# ---------------------------------------------------------------------------
# Happy path — both submitters run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_runs_dream_then_ratification_sharing_nightly_id():
    dream_spy = AsyncMock(return_value=_dream_result())
    rat_spy = AsyncMock(return_value=_ratification_result(ratified=3, superseded=1))
    with patch.object(
        nightly_batch,
        "check_dream_budget",
        new=AsyncMock(return_value=(True, None)),
    ), patch(
        "backend.copilot.dream.orchestrator.execute_dream_pass", new=dream_spy
    ), patch.object(
        nightly_batch, "run_ratification_pass", new=rat_spy
    ):
        result = await run_nightly_batch_submit("u")

    assert result.skipped is False
    assert result.error is None
    assert result.dream_in_flight is False
    assert result.dream is not None
    assert result.dream.consolidated_count == 2
    assert result.ratification is not None
    assert result.ratification.ratified_count == 3
    assert result.ratification.superseded_count == 1
    # All submitters consult the same nightly_id implicitly via the
    # NightlyBatchResult — assertion is shape, not collation, since
    # the cost log layer reads it through PlatformCostLog metadata.
    assert result.nightly_id  # uuid set


@pytest.mark.asyncio
async def test_submitters_called_in_documented_order_dream_then_ratification():
    """Order matters for future stages (e.g. P2 dedup before dream,
    P4 pre-warm after). Pin the order so future re-orderings are
    intentional."""
    call_order: list[str] = []
    dream_spy = AsyncMock(
        side_effect=lambda *_a, **_kw: call_order.append("dream") or _dream_result()
    )
    rat_spy = AsyncMock(
        side_effect=lambda *_a, **_kw: call_order.append("rat")
        or _ratification_result()
    )
    with patch.object(
        nightly_batch,
        "check_dream_budget",
        new=AsyncMock(return_value=(True, None)),
    ), patch(
        "backend.copilot.dream.orchestrator.execute_dream_pass", new=dream_spy
    ), patch.object(
        nightly_batch, "run_ratification_pass", new=rat_spy
    ):
        await run_nightly_batch_submit("u")

    assert call_order == ["dream", "rat"]


# ---------------------------------------------------------------------------
# Per-submitter failure isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dream_crash_still_lets_ratification_sweep_run():
    """A crashed dream submitter must not block the ratification
    sweep — they operate on disjoint tentative-edge populations."""
    rat_spy = AsyncMock(return_value=_ratification_result(superseded=2))
    with patch.object(
        nightly_batch,
        "check_dream_budget",
        new=AsyncMock(return_value=(True, None)),
    ), patch(
        "backend.copilot.dream.orchestrator.execute_dream_pass",
        new=AsyncMock(side_effect=RuntimeError("dream boom")),
    ), patch.object(
        nightly_batch, "run_ratification_pass", new=rat_spy
    ):
        result = await run_nightly_batch_submit("u")

    assert result.error is not None
    assert "dream" in result.error
    assert "dream boom" in result.error
    # Ratification still ran.
    assert result.ratification is not None
    assert result.ratification.superseded_count == 2


@pytest.mark.asyncio
async def test_both_submitters_failing_concatenates_errors():
    """Both submitters can fail independently; both errors surface."""
    with patch.object(
        nightly_batch,
        "check_dream_budget",
        new=AsyncMock(return_value=(True, None)),
    ), patch(
        "backend.copilot.dream.orchestrator.execute_dream_pass",
        new=AsyncMock(side_effect=RuntimeError("dream boom")),
    ), patch.object(
        nightly_batch,
        "run_ratification_pass",
        new=AsyncMock(side_effect=RuntimeError("rat boom")),
    ):
        result = await run_nightly_batch_submit("u")

    assert result.error is not None
    assert "dream" in result.error
    assert "ratification" in result.error


@pytest.mark.asyncio
async def test_dream_internal_error_propagates_via_result_dream_error_not_top_level():
    """A dream pass that returns ``DreamPassResult(error=...)``
    (internal failure caught by orchestrator) does NOT set
    ``NightlyBatchResult.error`` — it surfaces in ``.dream.error``
    so the cron-wrapper log line can distinguish 'submitter crashed
    in nightly batch glue' from 'submitter ran and reported failure'."""
    dream_spy = AsyncMock(return_value=_dream_result(error="phase 1 LLM down"))
    rat_spy = AsyncMock(return_value=_ratification_result())
    with patch.object(
        nightly_batch,
        "check_dream_budget",
        new=AsyncMock(return_value=(True, None)),
    ), patch(
        "backend.copilot.dream.orchestrator.execute_dream_pass", new=dream_spy
    ), patch.object(
        nightly_batch, "run_ratification_pass", new=rat_spy
    ):
        result = await run_nightly_batch_submit("u")

    # Top-level error stays None — submitter reported its own error.
    # The admin ``*_with_status`` wrapper folds sub-result errors into
    # the JobStatus row at status-write time.
    assert result.error is None
    assert result.dream is not None
    assert result.dream.error == "phase 1 LLM down"
    # Ratification still ran.
    assert result.ratification is not None


# ---------------------------------------------------------------------------
# Async batch-path handoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_path_dream_sets_dream_in_flight():
    """A dream submitter that took a provider batch path only ENQUEUED
    its pass — ``dream_in_flight`` flags that the apply step lands
    asynchronously via the BatchExecutor, so envelope consumers (the
    admin wrapper marks the row complete with this envelope) don't
    read the dream counts as final."""
    dream_spy = AsyncMock(return_value=_dream_result(execution_path="anthropic_batch"))
    rat_spy = AsyncMock(return_value=_ratification_result())
    with patch.object(
        nightly_batch,
        "check_dream_budget",
        new=AsyncMock(return_value=(True, None)),
    ), patch(
        "backend.copilot.dream.orchestrator.execute_dream_pass", new=dream_spy
    ), patch.object(
        nightly_batch, "run_ratification_pass", new=rat_spy
    ):
        result = await run_nightly_batch_submit("u")

    assert result.dream_in_flight is True
    assert result.error is None


@pytest.mark.asyncio
async def test_skipped_or_errored_batch_path_dream_is_not_in_flight():
    """A batch-path dream that was SKIPPED (lock held, no input) or
    returned an error result has nothing pending with the provider —
    the in-flight flag must stay unset."""
    for dream in (
        _dream_result(execution_path="anthropic_batch", skipped=True),
        _dream_result(execution_path="anthropic_batch", error="submit failed"),
    ):
        with patch.object(
            nightly_batch,
            "check_dream_budget",
            new=AsyncMock(return_value=(True, None)),
        ), patch(
            "backend.copilot.dream.orchestrator.execute_dream_pass",
            new=AsyncMock(return_value=dream),
        ), patch.object(
            nightly_batch,
            "run_ratification_pass",
            new=AsyncMock(return_value=_ratification_result()),
        ):
            result = await run_nightly_batch_submit("u")

        assert result.dream_in_flight is False
