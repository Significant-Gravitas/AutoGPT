"""Schema-level coverage for the custom Graphiti entity + edge types.

These tests pin the field defaults and constraints so a regression
on the audit-fix payload (status / confidence / scope / source_kind /
provenance) is caught locally before it round-trips through Graphiti's
LLM extraction step and ends up on production :RELATES_TO edges.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from .memory_model import MemoryStatus, SourceKind
from .types import (
    EDGE_TYPE_MAP,
    EDGE_TYPES,
    ENTITY_TYPES,
    MemoryFact,
    Organization,
    Person,
)


def test_memoryfact_defaults_match_envelope():
    """Defaults round-trip into a :RELATES_TO edge without surprises."""
    fact = MemoryFact()
    assert fact.status == MemoryStatus.active
    assert fact.confidence is None
    assert fact.source_kind == SourceKind.user_asserted
    assert fact.scope == "real:global"
    assert fact.provenance is None
    assert fact.web_verified_at is None
    assert fact.ratified_at is None
    assert fact.expiration_reason is None


@pytest.mark.parametrize(
    "value,ok",
    [(-0.01, False), (0.0, True), (0.5, True), (1.0, True), (1.01, False)],
)
def test_memoryfact_confidence_bounds(value, ok):
    if ok:
        MemoryFact(confidence=value)
    else:
        with pytest.raises(ValidationError):
            MemoryFact(confidence=value)


def test_memoryfact_status_enum_restricted():
    """Open enum on the envelope side, restricted on the edge side."""
    MemoryFact(status=MemoryStatus.tentative)
    with pytest.raises(ValidationError):
        MemoryFact(status="archived")  # type: ignore[arg-type]


def test_memoryfact_carries_full_envelope_metadata():
    """All audit-required fields can be set in one shot."""
    fact = MemoryFact(
        status=MemoryStatus.tentative,
        confidence=0.7,
        source_kind=SourceKind.assistant_derived,
        scope="project:foo",
        provenance="session:s1#msg:42",
        web_verified_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        ratified_at=datetime(2026, 5, 14, tzinfo=timezone.utc),
        expiration_reason="stale_fact",
    )
    dumped = fact.model_dump(mode="json")
    assert dumped["status"] == "tentative"
    assert dumped["confidence"] == 0.7
    assert dumped["source_kind"] == "assistant_derived"
    assert dumped["scope"] == "project:foo"
    assert dumped["provenance"] == "session:s1#msg:42"
    assert dumped["web_verified_at"] == "2026-05-12T00:00:00Z"
    assert dumped["ratified_at"] == "2026-05-14T00:00:00Z"
    assert dumped["expiration_reason"] == "stale_fact"


def test_entity_types_include_the_six_canonical_kinds():
    """Audit §6.3 — narrow entity allowlist."""
    expected = {"Person", "Organization", "Project", "Concept", "Preference", "Rule"}
    assert expected.issubset(set(ENTITY_TYPES.keys()))


def test_edge_types_include_memoryfact():
    assert "MemoryFact" in EDGE_TYPES
    assert EDGE_TYPES["MemoryFact"] is MemoryFact


def test_edge_type_map_spans_full_cartesian_product():
    """Every (entity, entity) pair maps to MemoryFact — the audit's
    'one edge type, narrow entity vocabulary' shape."""
    # Six entity types × six entity types = 36 pairs.
    assert len(EDGE_TYPE_MAP) == len(ENTITY_TYPES) ** 2
    for src in ENTITY_TYPES:
        for tgt in ENTITY_TYPES:
            assert EDGE_TYPE_MAP.get((src, tgt)) == ["MemoryFact"]


def test_person_entity_has_optional_role_and_email():
    p = Person()
    assert p.role is None
    assert p.email is None
    p = Person(role="CEO", email="ceo@acme.example")
    assert p.role == "CEO"
    assert p.email == "ceo@acme.example"


def test_organization_entity_has_optional_industry():
    o = Organization()
    assert o.industry is None
    o = Organization(industry="aerospace")
    assert o.industry == "aerospace"
