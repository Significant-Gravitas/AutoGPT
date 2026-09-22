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


# ---------------------------------------------------------------------------
# RecombinationOutput lenient parsing
# ---------------------------------------------------------------------------


def test_recombination_output_drops_proposals_with_unknown_memory_kind():
    """LLMs sometimes invent kinds outside MemoryKind (e.g. 'inference',
    'meta'). Those proposals are dropped; valid ones are kept."""
    from .schemas import ProposedFinding, RecombinationOutput

    payload = {
        "proposals": [
            {
                "content": "valid one",
                "scope": "real:global",
                "memory_kind": "finding",
                "confidence": 0.8,
                "rationale": "ok",
            },
            {
                "content": "unknown kind",
                "scope": "real:global",
                "memory_kind": "inference",  # not in enum
                "confidence": 0.5,
                "rationale": "should be dropped",
            },
            {
                "content": "another bad kind",
                "scope": "real:global",
                "memory_kind": "meta",  # not in enum
                "confidence": 0.5,
                "rationale": "should be dropped",
            },
        ]
    }
    out = RecombinationOutput.model_validate(payload)
    assert len(out.proposals) == 1
    assert isinstance(out.proposals[0], ProposedFinding)
    assert out.proposals[0].content == "valid one"


def test_recombination_output_keeps_proposals_without_memory_kind():
    """memory_kind has a default of MemoryKind.finding, so omitting it
    is fine — proposal must still be kept."""
    from .schemas import RecombinationOutput

    payload = {
        "proposals": [
            {
                "content": "no kind given",
                "scope": "real:global",
                "confidence": 0.7,
                "rationale": "default kind applies",
            }
        ]
    }
    out = RecombinationOutput.model_validate(payload)
    assert len(out.proposals) == 1
    # default should be MemoryKind.finding
    assert out.proposals[0].memory_kind.value == "finding"


def test_recombination_output_all_invalid_yields_empty_list():
    """If every proposal has an invalid memory_kind the result is an empty
    proposals list — phase 3 then has nothing to gate, which is fine."""
    from .schemas import RecombinationOutput

    payload = {
        "proposals": [
            {
                "content": "a",
                "memory_kind": "metaphysics",
                "confidence": 0.5,
                "rationale": "drop me",
            },
            {
                "content": "b",
                "memory_kind": "speculation",
                "confidence": 0.5,
                "rationale": "drop me too",
            },
        ]
    }
    out = RecombinationOutput.model_validate(payload)
    assert out.proposals == []
