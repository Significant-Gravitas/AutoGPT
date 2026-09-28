"""The recall guard at clamp time (``clamp.clamp_pass_operations``): a
protected demotion is dropped before the cap slice, so it never takes the slot
of a demotion apply would write, and each drop is counted."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from backend.copilot.graphiti.recall_stamp import stamp_time

from . import recall_guard
from .clamp import clamp_operations, clamp_pass_operations
from .fetch import DreamInput, FactRow
from .schemas import DreamDemotion, DreamOperations


def _fact(uuid: str, *, recalled_days_ago: float | None = None) -> FactRow:
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


def _ops(*demotions: DreamDemotion) -> DreamOperations:
    return DreamOperations(demotions=list(demotions), summary_for_user="ok")


def _demote(uuid: str, reason: str = "stale_fact") -> DreamDemotion:
    return DreamDemotion(edge_uuid=uuid, reason=reason)


@pytest.fixture
def window(mocker):
    def _set(days: int) -> None:
        mocker.patch.object(
            recall_guard,
            "Settings",
            return_value=SimpleNamespace(
                config=SimpleNamespace(dream_demotion_protect_days=days)
            ),
        )

    return _set


def test_a_protected_demotion_does_not_take_the_only_slot() -> None:
    """A small graph's cap floors at one slot. Unguarded, the protected
    demotion at the head of the list takes it and the stale fact survives
    another night; guarded, the stale fact is demoted."""
    facts = [_fact("hot", recalled_days_ago=1), _fact("cold")] + [
        _fact(f"f{i}") for i in range(8)
    ]
    ops = _ops(_demote("hot"), _demote("cold"))

    clamped = clamp_pass_operations(ops, _bundle(facts))

    assert [d.edge_uuid for d in clamped.ops.demotions] == ["cold"]
    assert clamped.protected_demotions == 1
    assert [d.edge_uuid for d in clamp_operations(ops, len(facts)).demotions] == ["hot"]


def test_the_cap_is_filled_by_unprotected_demotions() -> None:
    hot = [_fact(f"hot{i}", recalled_days_ago=i + 1) for i in range(3)]
    cold = [_fact(f"cold{i}") for i in range(7)]
    rest = [_fact(f"f{i}") for i in range(90)]
    ops = _ops(*(_demote(f.uuid) for f in hot + cold))

    clamped = clamp_pass_operations(ops, _bundle(hot + cold + rest))

    # 100 active facts: a cap of five, all of them unprotected.
    assert [d.edge_uuid for d in clamped.ops.demotions] == [
        f"cold{i}" for i in range(5)
    ]
    assert clamped.protected_demotions == 3


def test_a_contradiction_or_a_retraction_still_demotes_a_protected_fact() -> None:
    facts = [
        _fact("hot", recalled_days_ago=1),
        _fact("witness"),
        *(_fact(f"f{i}") for i in range(98)),
    ]
    ops = _ops(
        _demote("hot", "user_signal"),
        _demote("hot", "contradicted_by:witness"),
        _demote("hot", "contradicted_by:hot"),
        _demote("hot", "contradicted_by:never-read"),
        _demote("hot", "web_contradicted:https://example.test"),
        _demote("hot", "stale_fact"),
    )

    clamped = clamp_pass_operations(ops, _bundle(facts))

    assert [d.reason for d in clamped.ops.demotions] == [
        "user_signal",
        "contradicted_by:witness",
    ]
    assert clamped.protected_demotions == 4


def test_a_fact_recalled_outside_the_window_is_not_protected(window) -> None:
    window(7)
    facts = [
        _fact("recent", recalled_days_ago=3),
        _fact("older", recalled_days_ago=10),
        *(_fact(f"f{i}") for i in range(98)),
    ]
    ops = _ops(_demote("recent"), _demote("older"))

    clamped = clamp_pass_operations(ops, _bundle(facts))

    assert [d.edge_uuid for d in clamped.ops.demotions] == ["older"]
    assert clamped.protected_demotions == 1


def test_a_zero_window_turns_the_clamp_time_guard_off(window) -> None:
    window(0)
    facts = [_fact("hot", recalled_days_ago=0.01), *(_fact(f"f{i}") for i in range(99))]
    ops = _ops(_demote("hot"))

    clamped = clamp_pass_operations(ops, _bundle(facts))

    assert [d.edge_uuid for d in clamped.ops.demotions] == ["hot"]
    assert clamped.protected_demotions == 0


def test_the_rest_of_the_clamp_is_unchanged() -> None:
    """Everything but the demotions is what ``clamp_operations`` makes of it,
    and a demotion of a fact the pass never read is still dropped uncounted."""
    facts = [_fact(f"f{i}") for i in range(100)]
    ops = DreamOperations(
        demotions=[_demote("hallucinated"), _demote("f1")], summary_for_user="done"
    )

    clamped = clamp_pass_operations(ops, _bundle(facts))

    assert clamped.ops == clamp_operations(
        ops, 100, known_fact_uuids={f.uuid for f in facts}
    )
    assert [d.edge_uuid for d in clamped.ops.demotions] == ["f1"]
    assert clamped.protected_demotions == 0
