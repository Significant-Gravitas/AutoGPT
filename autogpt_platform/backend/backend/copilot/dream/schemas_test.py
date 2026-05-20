"""Schema validation — bounds and defaults for the dream output models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from .schemas import (
    ConsolidatedFact,
    DreamDemotion,
    DreamOperations,
    DreamPassResult,
    ProposedFinding,
)


def test_consolidated_fact_confidence_clamped_to_unit_interval():
    ConsolidatedFact(content="x", confidence=0.0)
    ConsolidatedFact(content="x", confidence=1.0)
    with pytest.raises(ValidationError):
        ConsolidatedFact(content="x", confidence=-0.1)
    with pytest.raises(ValidationError):
        ConsolidatedFact(content="x", confidence=1.1)


def test_proposed_finding_requires_rationale():
    # ``rationale`` is required — phase 2 prompt insists on it and
    # phase 3 sanitizer rejects citations-less proposals; the schema
    # enforces that the field is present even if "" so callers can't
    # accidentally drop it.
    with pytest.raises(ValidationError):
        ProposedFinding(content="x", confidence=0.5)  # type: ignore[call-arg]


def test_demotion_status_must_be_one_of_two_values():
    DreamDemotion(edge_uuid="u", reason="r", new_status="superseded")
    DreamDemotion(edge_uuid="u", reason="r", new_status="contradicted")
    with pytest.raises(ValidationError):
        DreamDemotion(
            edge_uuid="u", reason="r", new_status="archived"  # type: ignore[arg-type]
        )


def test_dream_operations_defaults_to_empty_no_op():
    ops = DreamOperations()
    assert ops.writes == []
    assert ops.proposals == []
    assert ops.demotions == []
    assert ops.entity_invalidations == []
    assert ops.summary_for_user == ""


def test_dream_pass_result_skipped_keeps_zero_counts():
    result = DreamPassResult(
        user_id="u", pass_id="p", skipped=True, skip_reason="no_input"
    )
    assert result.consolidated_count == 0
    assert result.proposal_count == 0
    assert result.demotion_count == 0
    assert result.entity_invalidation_count == 0
    assert result.execution_path == "sync_baseline"
