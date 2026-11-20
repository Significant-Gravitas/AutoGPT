"""Usage data never changes which operations a pass attempts: the clamp both
routes call (``clamp.clamp_pass_operations``) selects from the proposals and
the fact count alone. Dropping a recently recalled fact's demotion before the
cap would hand its slot to another target, and the pass could then demote
more than it would without usage data (#13776's first redo did, 1 -> 2)."""

import random
from datetime import datetime, timedelta, timezone

from backend.copilot.graphiti.recall_stamp import stamp_time

from .clamp import clamp_operations, clamp_pass_operations
from .fetch import DreamInput, FactRow
from .schemas import DreamDemotion, DreamOperations, EntityInvalidation


def _fact(uuid: str, recalled_days_ago: float | None = None) -> FactRow:
    last = (
        stamp_time(datetime.now(timezone.utc) - timedelta(days=recalled_days_ago))
        if recalled_days_ago is not None
        else None
    )
    return FactRow(
        uuid=uuid,
        source="A",
        target="B",
        name="likes",
        fact=f"fact {uuid}",
        scope="real:global",
        confidence=0.8,
        status="active",
        created_at="2026-01-01T00:00:00+00:00",
        recall_count=1 if last else None,
        last_recalled_at=last,
        prev_recalled_at=last,
    )


def _bundle(facts: list[FactRow]) -> DreamInput:
    now = datetime.now(timezone.utc)
    return DreamInput(
        user_id="u",
        group_id="g",
        window_start=now,
        window_end=now,
        facts=facts,
        known_fact_uuids={f.uuid for f in facts},
    )


def _without_usage(facts: list[FactRow]) -> list[FactRow]:
    return [
        f.model_copy(
            update={
                "recall_count": None,
                "last_recalled_at": None,
                "prev_recalled_at": None,
            }
        )
        for f in facts
    ]


def test_a_recalled_fact_keeps_its_cap_slot_and_nothing_is_backfilled() -> None:
    """Codex's case A: forty facts, a cap of two, stale proposals
    ``[A, A, B, C]`` with A recalled yesterday. Both A proposals keep their
    slots; B and C are not attempted in either world."""
    facts = [_fact("A", 1), _fact("B"), _fact("C")] + [
        _fact(f"filler-{i}") for i in range(37)
    ]
    ops = DreamOperations(
        demotions=[
            DreamDemotion(edge_uuid=uuid, reason="stale_fact")
            for uuid in ("A", "A", "B", "C")
        ],
        summary_for_user="x",
    )

    with_usage = clamp_pass_operations(ops, _bundle(facts))
    without = clamp_pass_operations(ops, _bundle(_without_usage(facts)))

    assert [d.edge_uuid for d in with_usage.demotions] == ["A", "A"]
    assert with_usage == without


def test_the_selection_is_the_same_whatever_the_usage_history() -> None:
    rng = random.Random("clamp-usage")
    for _ in range(200):
        uuids = [f"f{i}" for i in range(rng.randint(1, 120))]
        facts = [
            _fact(uuid, rng.choice([None, 0.01, 1, 20, 29, 31, 40, 90, 400]))
            for uuid in uuids
        ]
        ops = DreamOperations(
            demotions=[
                DreamDemotion(
                    edge_uuid=rng.choice(uuids + ["hallucinated"]),
                    reason=rng.choice(
                        ["stale_fact", "user_signal", f"contradicted_by:{uuids[0]}"]
                    ),
                )
                for _ in range(rng.randint(0, 15))
            ],
            entity_invalidations=[
                EntityInvalidation(entity_uuid=f"e{i}", reason="stale_fact")
                for i in range(rng.randint(0, 4))
            ],
            summary_for_user="x",
        )

        with_usage = clamp_pass_operations(ops, _bundle(facts))

        assert with_usage == clamp_pass_operations(ops, _bundle(_without_usage(facts)))
        assert with_usage == clamp_operations(
            ops, len(facts), known_fact_uuids=set(uuids)
        )
