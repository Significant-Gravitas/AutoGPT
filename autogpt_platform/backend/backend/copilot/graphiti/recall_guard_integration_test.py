"""The recall guard in the dream's destructive writes, on a live FalkorDB.

Each write (``guarded_writes.py``) tests the recall stamps in the statement
that writes (``recall_stamp.spared_by_recall``): ``supersede_unless_recalled``
for a demotion or the ratification sweep's supersession, and
``invalidate_entity_direct_neighbors`` for an entity's neighbours, change a
live fact unless it was recalled within the window and the write's override
does not reach it, and return what they changed and what they spared. This
file runs those statements; proves the stamps order by time as the strings
``stamp_time`` writes them; and keeps the live regressions from Codex's first
validation of this PR: two passes that demoted more with usage data than
without it (cases A and B), and a recall stamped between the pass's read of
the graph and its write, on each writer.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_guard_integration_test.py
"""

from datetime import datetime, timedelta, timezone
from itertools import product
from types import SimpleNamespace
from typing import Any

import pytest

from backend.copilot.dream import apply as apply_mod
from backend.copilot.dream import demotions
from backend.copilot.dream.clamp import clamp_pass_operations
from backend.copilot.dream.fetch import DreamInput, _fetch_active_facts
from backend.copilot.dream.schemas import (
    DreamDemotion,
    DreamOperations,
    EntityInvalidation,
)

from .falkordb_driver import AutoGPTFalkorDriver
from .guarded_writes import (
    WriteOutcome,
    invalidate_entity_direct_neighbors,
    supersede_unless_recalled,
)
from .recall_integration_fixtures import edge_row, rows
from .recall_stamp import RecallProtection, parse_stamp, stamp_recalls, stamp_time

_OWNER = "u-guard-integration"
CHANGED, SPARED, FAILED = WriteOutcome.CHANGED, WriteOutcome.SPARED, WriteOutcome.FAILED
_UTC_PLUS_10 = timezone(timedelta(hours=10))
_UTC_MINUS_5 = timezone(timedelta(hours=-5))
# Each digit rollover of a stamp, and two local times whose own ISO strings
# order the wrong way round (09:00+10:00 is 23:00Z the day before
# 20:00-05:00, which is 01:00Z): ``stamp_time`` writes them in UTC.
_MOMENTS = [
    datetime(2026, 9, 28, 9, 59, 59, 999999, tzinfo=timezone.utc),
    datetime(2026, 9, 28, 10, 0, tzinfo=timezone.utc),
    datetime(2026, 9, 30, 23, 59, 59, 999999, tzinfo=timezone.utc),
    datetime(2026, 10, 1, tzinfo=timezone.utc),
    datetime(2026, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc),
    datetime(2027, 1, 1, tzinfo=timezone.utc),
    datetime(2026, 9, 28, 9, 0, tzinfo=_UTC_PLUS_10),
    datetime(2026, 9, 27, 20, 0, tzinfo=_UTC_MINUS_5),
]


def _ago(**kwargs: float) -> str:
    return stamp_time(datetime.now(timezone.utc) - timedelta(**kwargs))


def _window(**kwargs: Any) -> RecallProtection:
    """The default 30-day window, and an override if given."""
    return RecallProtection(recalled_since=_ago(days=30), **kwargs)


async def _edge(
    driver: AutoGPTFalkorDriver,
    group_id: str,
    uuid: str,
    *,
    source: str | None = None,
    target: str | None = None,
    **props: Any,
) -> None:
    """A live fact from *source* to *target* (each entity made on first
    use), with *props* set on it."""
    await driver.execute_query(
        """
        MERGE (s:Entity {uuid: $source, group_id: $g})
          ON CREATE SET s.name = $source
        MERGE (t:Entity {uuid: $target, group_id: $g})
          ON CREATE SET t.name = $target
        CREATE (s)-[e:RELATES_TO {uuid: $uuid, group_id: $g, name: 'knows',
                                  fact: $uuid + ' is known', status: 'active',
                                  scope: 'real:global', confidence: 0.9,
                                  created_at: $created}]->(t)
        SET e += $props
        """,
        uuid=uuid,
        g=group_id,
        source=source or f"{uuid}-src",
        target=target or f"{uuid}-tgt",
        created=_ago(days=100),
        props=props,
    )


async def _statuses(driver: AutoGPTFalkorDriver, uuids: list[str]) -> dict[str, str]:
    return {uuid: (await edge_row(driver, uuid))["status"] for uuid in uuids}


async def _supersede(
    driver: AutoGPTFalkorDriver, group_id: str, uuid: str, protection: RecallProtection
) -> WriteOutcome:
    [outcome] = await supersede_unless_recalled(
        driver,
        [uuid],
        reason="stale_fact",
        new_status="superseded",
        group_id=group_id,
        protection=protection,
    )
    return outcome


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "recalled_days, outcome",
    [(None, CHANGED), (1, SPARED), (20, SPARED), (29, SPARED), (40, CHANGED)],
    ids=["never", "yesterday", "20-days", "29-days", "40-days"],
)
async def test_the_window_decides_when_no_override_reaches_the_fact(
    clean_graph, recalled_days: float | None, outcome: WriteOutcome
) -> None:
    driver, group_id = clean_graph
    stamps = (
        {} if recalled_days is None else {"last_recalled_at": _ago(days=recalled_days)}
    )
    await _edge(driver, group_id, "fact", **stamps)

    assert await _supersede(driver, group_id, "fact", _window()) == outcome

    status = "active" if outcome == SPARED else "superseded"
    assert await _statuses(driver, ["fact"]) == {"fact": status}


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "protection, outcome",
    [
        (RecallProtection(), CHANGED),
        (_window(override=True), CHANGED),
        (_window(override=True, cited="other"), CHANGED),
        (_window(override=True, cited="fact"), SPARED),
        (_window(cited="ghost"), SPARED),
    ],
    ids=[
        "window-off",
        "user-signal",
        "contradicted-by-another",
        "contradicted-by-itself",
        "unknown-citation",
    ],
)
async def test_an_override_reaches_a_recent_recall_but_not_its_own_citation(
    clean_graph, protection: RecallProtection, outcome: WriteOutcome
) -> None:
    driver, group_id = clean_graph
    await _edge(driver, group_id, "fact", last_recalled_at=_ago(days=1))

    assert await _supersede(driver, group_id, "fact", protection) == outcome


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "props",
    [
        {"status": "superseded", "expired_at": _ago(days=2)},
        {"forgotten_at": _ago(days=2)},
        {"status": "retracted"},
    ],
    ids=["demoted", "forgotten", "retracted"],
)
async def test_a_fact_no_longer_live_fails_and_is_left_as_it_is(
    clean_graph, props: dict[str, Any]
) -> None:
    driver, group_id = clean_graph
    await _edge(driver, group_id, "gone", last_recalled_at=_ago(days=90), **props)
    before = await edge_row(driver, "gone")

    assert await _supersede(driver, group_id, "gone", _window()) == FAILED
    assert await _supersede(driver, group_id, "missing", _window()) == FAILED

    assert await edge_row(driver, "gone") == before


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, recalled_days, outcome, after",
    [
        ("tentative", None, CHANGED, "superseded"),
        ("tentative", 1, SPARED, "tentative"),
        ("active", None, FAILED, "active"),
    ],
    ids=["tentative", "tentative-recalled", "promoted-since-the-listing"],
)
async def test_the_ratification_write_supersedes_only_a_still_tentative_fact(
    clean_graph,
    status: str,
    recalled_days: float | None,
    outcome: WriteOutcome,
    after: str,
) -> None:
    """The sweep's supersession: ``expected_status`` keeps it off a proposal
    promoted since the sweep listed it, and with no override a recent recall
    spares a proposal whose Redis hit count was lost."""
    driver, group_id = clean_graph
    stamps = (
        {} if recalled_days is None else {"last_recalled_at": _ago(days=recalled_days)}
    )
    await _edge(driver, group_id, "proposal", status=status, **stamps)

    [written] = await supersede_unless_recalled(
        driver,
        ["proposal"],
        reason="unratified",
        new_status="superseded",
        group_id=group_id,
        protection=_window(),
        expected_status="tentative",
    )

    assert written == outcome
    assert await _statuses(driver, ["proposal"]) == {"proposal": after}


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "protection, spared",
    [
        (_window(), {"fresh", "fresh-too", "loop"}),
        (_window(override=True), set()),
        (_window(override=True, cited="fresh"), {"fresh"}),
        (RecallProtection(), set()),
    ],
    ids=["window", "user-signal", "contradicted-by-fresh", "window-off"],
)
async def test_the_neighbour_write_spares_and_reports_edge_by_edge(
    clean_graph, protection: RecallProtection, spared: set[str]
) -> None:
    """A self-loop is one neighbour, not two; a neighbour no longer live is
    in neither list and keeps its status."""
    driver, group_id = clean_graph
    await _edge(driver, group_id, "fresh", source="hub", last_recalled_at=_ago(days=1))
    await _edge(
        driver, group_id, "fresh-too", target="hub", last_recalled_at=_ago(days=5)
    )
    await _edge(
        driver,
        group_id,
        "loop",
        source="hub",
        target="hub",
        last_recalled_at=_ago(days=1),
    )
    await _edge(driver, group_id, "old", source="hub", last_recalled_at=_ago(days=40))
    await _edge(driver, group_id, "never", target="hub")
    await _edge(driver, group_id, "expired", source="hub", status="contradicted")
    live = {"fresh", "fresh-too", "loop", "old", "never"}

    writes = await invalidate_entity_direct_neighbors(
        driver, group_id, "hub", "stale_fact", protection=protection
    )

    assert sorted(writes.spared) == sorted(spared)
    assert sorted(writes.changed) == sorted(live - spared)
    statuses = await _statuses(driver, [*live, "expired"])
    assert statuses == {
        **{uuid: "active" for uuid in spared},
        **{uuid: "superseded" for uuid in live - spared},
        "expired": "contradicted",
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_stamp_is_stored_as_the_string_stamp_time_writes(clean_graph) -> None:
    """A real stamp is a string, exactly ``stamp_time``'s output, and the
    guard compares it to the microsecond: a cutoff equal to it spares, one a
    microsecond later does not."""
    driver, group_id = clean_graph
    await _edge(driver, group_id, "at")
    await _edge(driver, group_id, "before")
    assert await stamp_recalls(driver, ["at", "before"], owner=_OWNER) == 2
    [row] = await rows(
        driver,
        "MATCH ()-[e:RELATES_TO {uuid: 'at'}]->() RETURN e.last_recalled_at AS stamp",
    )
    stored = row["stamp"]
    moment = parse_stamp(stored)

    assert moment is not None and stamp_time(moment) == stored
    later = stamp_time(moment + timedelta(microseconds=1))
    assert (
        await _supersede(
            driver, group_id, "at", RecallProtection(recalled_since=stored)
        )
        == SPARED
    )
    assert (
        await _supersede(
            driver, group_id, "before", RecallProtection(recalled_since=later)
        )
        == CHANGED
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stamps_order_by_time_across_every_rollover_and_zone(clean_graph) -> None:
    """Every pair of ``_MOMENTS``, one as the stamp and one as the cutoff:
    FalkorDB's string comparison spares exactly when the stamp's time is at
    or after the cutoff's."""
    driver, group_id = clean_graph
    pairs = list(product(enumerate(_MOMENTS), repeat=2))
    for (i, stamp), (j, _) in pairs:
        await _edge(driver, group_id, f"p{i}-{j}", last_recalled_at=stamp_time(stamp))

    outcomes = {
        (i, j): await _supersede(
            driver,
            group_id,
            f"p{i}-{j}",
            RecallProtection(recalled_since=stamp_time(cutoff)),
        )
        for (i, _), (j, cutoff) in pairs
    }

    assert outcomes == {
        (i, j): SPARED if stamp >= cutoff else CHANGED
        for (i, stamp), (j, cutoff) in pairs
    }


class _StampBeforeWrite:
    """The pass's driver; with *marker*, a real recall of ``A`` is stamped
    just before the first query whose Cypher contains it runs: after the
    pass read the graph, before its write."""

    def __init__(self, driver: AutoGPTFalkorDriver, marker: str | None) -> None:
        self.driver = driver
        self.marker = marker
        self.stamped = 0

    async def execute_query(self, query: str, **params: Any) -> Any:
        if self.marker and self.marker in query and not self.stamped:
            self.stamped = await stamp_recalls(self.driver, ["A"], owner=_OWNER)
        return await self.driver.execute_query(query, **params)

    async def close(self) -> None:
        """The fixture closes the driver."""


@pytest.fixture
def live_pass(mocker, scope_graph) -> SimpleNamespace:
    """A dream pass on the live graph through the production gather, clamp
    and apply; only the chat store and the entity flag (on) are stubbed."""
    driver, scope = scope_graph
    mocker.patch.object(apply_mod, "_create_dream_session", return_value="s")
    mocker.patch.object(apply_mod, "_write_dream_summary_message")
    mocker.patch.object(demotions, "is_feature_enabled", return_value=True)
    state = SimpleNamespace(driver=driver, scope=scope, boundary=None)

    async def run(ops: DreamOperations, stamp_before: str | None = None) -> dict:
        state.boundary = _StampBeforeWrite(driver, stamp_before)
        mocker.patch.object(demotions, "open_driver", return_value=state.boundary)
        facts = await _fetch_active_facts(driver, scope.group_id, 100)
        now = datetime.now(timezone.utc)
        bundle = DreamInput(
            user_id=scope.owner_user_id,
            group_id=scope.group_id,
            window_start=now - timedelta(days=14),
            window_end=now,
            facts=facts,
            known_fact_uuids={f.uuid for f in facts},
        )
        return await apply_mod.apply_operations(
            scope,
            "p-guard",
            clamp_pass_operations(ops, bundle),
            known_fact_uuids=bundle.known_fact_uuids,
        )

    state.run = run
    return state


def _stale(*uuids: str) -> list[DreamDemotion]:
    return [DreamDemotion(edge_uuid=uuid, reason="stale_fact") for uuid in uuids]


def _changed(stats: dict) -> int:
    return int(stats["demotion_count"]) + int(stats["entity_invalidation_count"])


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "recalled, changed, protected, a_status",
    [(False, 1, 0, "superseded"), (True, 0, 1, "active")],
    ids=["no-usage", "with-usage"],
)
async def test_case_a_a_recalled_fact_keeps_its_cap_slots(
    live_pass, recalled: bool, changed: int, protected: int, a_status: str
) -> None:
    """Forty facts, a cap of two, stale proposals ``[A, A, B, C]``. Without
    usage the first write demotes A and the second finds it gone; with A
    recalled both writes spare it, and it counts once as protected. B and C
    are attempted in neither world: the first redo of this PR dropped A's
    proposals at the cap and demoted B and C instead (1 -> 2)."""
    driver, scope = live_pass.driver, live_pass.scope
    for uuid in ["A", "B", "C", *(f"filler-{i}" for i in range(37))]:
        await _edge(driver, scope.group_id, uuid)
    if recalled:
        assert await stamp_recalls(driver, ["A"], owner=_OWNER) == 1

    stats = await live_pass.run(DreamOperations(demotions=_stale("A", "A", "B", "C")))

    assert (_changed(stats), stats["protected_demotions"]) == (changed, protected)
    assert await _statuses(driver, ["A", "B", "C"]) == {
        "A": a_status,
        "B": "active",
        "C": "active",
    }


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "recalled, protected", [(False, 0), (True, 1)], ids=["no-usage", "with-usage"]
)
async def test_case_b_a_user_retraction_through_an_entity_changes_no_more(
    live_pass, recalled: bool, protected: int
) -> None:
    """Two facts, a cap of one, stale proposals ``[A, B]`` and a
    ``user_signal`` invalidation of the entity only A hangs off. Without
    usage the direct write demotes A and the entity finds it gone; with A
    recalled the direct write spares it and the user's retraction demotes it
    through the entity. One edge changes either way: the first redo demoted
    B as well (1 -> 2)."""
    driver, scope = live_pass.driver, live_pass.scope
    await _edge(driver, scope.group_id, "A", source="hub")
    await _edge(driver, scope.group_id, "B")
    if recalled:
        assert await stamp_recalls(driver, ["A"], owner=_OWNER) == 1

    stats = await live_pass.run(
        DreamOperations(
            demotions=_stale("A", "B"),
            entity_invalidations=[
                EntityInvalidation(entity_uuid="hub", reason="user_signal")
            ],
        )
    )

    assert (_changed(stats), stats["protected_demotions"]) == (1, protected)
    assert await _statuses(driver, ["A", "B"]) == {"A": "superseded", "B": "active"}


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ops, marker",
    [
        (DreamOperations(demotions=_stale("A")), "SET e.expired_at"),
        (
            DreamOperations(
                entity_invalidations=[
                    EntityInvalidation(entity_uuid="hub", reason="stale_fact")
                ]
            ),
            "SET r.expired_at",
        ),
    ],
    ids=["direct", "neighbour"],
)
async def test_a_recall_stamped_between_the_read_and_the_write_protects(
    live_pass, ops: DreamOperations, marker: str
) -> None:
    """A was never recalled when the pass read the graph, and is recalled
    just before the write runs. The first redo read the stamps before its
    writes and demoted A; the write's own test sees the stamp."""
    driver, scope = live_pass.driver, live_pass.scope
    await _edge(driver, scope.group_id, "A", source="hub")
    await _edge(driver, scope.group_id, "B")

    stats = await live_pass.run(ops, stamp_before=marker)

    assert live_pass.boundary.stamped == 1, "the stamp landed before the write"
    assert (_changed(stats), stats["protected_demotions"]) == (0, 1)
    assert await _statuses(driver, ["A", "B"]) == {"A": "active", "B": "active"}
