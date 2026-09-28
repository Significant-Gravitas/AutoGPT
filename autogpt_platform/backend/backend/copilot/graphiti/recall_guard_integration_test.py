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
the graph and its write, on each writer. Its second validation added the
rest: a hub above FalkorDB's 10,000-row result limit is accounted in full,
and a write whose acknowledgement is lost after it committed is reported
indeterminate and never leaves a false protected count. Its third added a
write still queued behind another when the pass's final read runs, which
the read overtakes and which lands after the pass returned: any write of
unknown outcome leaves the count provisional.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_guard_integration_test.py
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from itertools import product
from types import SimpleNamespace
from typing import Any

import pytest
from redis.asyncio import Redis

from backend.copilot.dream import apply as apply_mod
from backend.copilot.dream import demotions
from backend.copilot.dream.clamp import clamp_pass_operations
from backend.copilot.dream.fetch import DreamInput, _fetch_active_facts
from backend.copilot.dream.schemas import (
    DreamDemotion,
    DreamOperations,
    EntityInvalidation,
)

from .config import graphiti_config
from .falkordb_driver import AutoGPTFalkorDriver
from .guarded_writes import (
    WriteOutcome,
    invalidate_entity_direct_neighbors,
    supersede_unless_recalled,
)
from .recall_integration_fixtures import edge_row, rows
from .recall_stamp import RecallProtection, parse_stamp, stamp_recalls, stamp_time

_OWNER = "u-guard-integration"
CHANGED, SPARED = WriteOutcome.CHANGED, WriteOutcome.SPARED
UNMATCHED, UNKNOWN = WriteOutcome.UNMATCHED, WriteOutcome.UNKNOWN
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

    assert await _supersede(driver, group_id, "gone", _window()) == UNMATCHED
    assert await _supersede(driver, group_id, "missing", _window()) == UNMATCHED

    assert await edge_row(driver, "gone") == before


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, recalled_days, outcome, after",
    [
        ("tentative", None, CHANGED, "superseded"),
        ("tentative", 1, SPARED, "tentative"),
        ("active", None, UNMATCHED, "active"),
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


class _Boundary:
    """The pass's driver, and what a real server or network can do to it.

    With *stamp_before*, a real recall of ``A`` is stamped just before the
    first query whose Cypher contains it runs: after the pass read the
    graph, before its write. With *lose_reason*, a write made for that
    reason commits and then raises, as if its reply were lost. With
    *fail_read*, the stage's final liveness read raises."""

    def __init__(
        self,
        driver: AutoGPTFalkorDriver,
        stamp_before: str | None = None,
        lose_reason: str | None = None,
        fail_read: bool = False,
    ) -> None:
        self.driver = driver
        self.stamp_before = stamp_before
        self.lose_reason = lose_reason
        self.fail_read = fail_read
        self.stamped = 0
        self.lost_replies = 0

    async def execute_query(self, query: str, **params: Any) -> Any:
        if self.fail_read and "AS live" in query:
            raise TimeoutError("the liveness read's reply was lost")
        if self.stamp_before and self.stamp_before in query and not self.stamped:
            self.stamped = await stamp_recalls(self.driver, ["A"], owner=_OWNER)
        reply = await self.driver.execute_query(query, **params)
        if self.lose_reason and params.get("reason") == self.lose_reason:
            self.lost_replies += 1
            raise TimeoutError("committed, but the reply was lost")
        return reply

    async def close(self) -> None:
        """The fixture closes the driver."""


# A write on the hub that holds FalkorDB's writer until the server's clock is
# $hold_ms past the write's start, then sets a property; reads still run
# beside it. It reads the clock on every row it counts (``0 * j`` keeps the
# planner from testing it once, before the count), so how long it holds does
# not depend on the machine's speed. The count is capped, and it holds no
# more memory than two short lists.
_BLOCKER = (
    "MATCH (n:Entity {uuid: 'hub'}) "
    "WITH n, timestamp() + $hold_ms AS until "
    "UNWIND range(1, 10000) AS i "
    "UNWIND range(1, 100000) AS j "
    "WITH n, until WHERE timestamp() + 0 * j >= until "
    "WITH n, until LIMIT 1 "
    "SET n.barrier = until RETURN until // holds the writer"
)
# The writer is held at least this long, and ten times as long as the stage
# had run when the retraction arrived, so a slower machine waits longer.
_HOLD_SECONDS = 5.5
_HOLD_FACTOR = 10


class _Queued(_Boundary):
    """The pass's driver, with its write for *reason* delivered but still
    queued on the server when the pass goes on: a long write on the same
    graph holds FalkorDB's writer, the statement is sent behind it on a
    connection of its own, and the pass's call raises as if its connection
    had dropped. The pass's final read, a ``GRAPH.RO_QUERY``, does not wait
    for queued writes."""

    def __init__(
        self, driver: AutoGPTFalkorDriver, redis: Redis, group_id: str, reason: str
    ) -> None:
        super().__init__(driver)
        self.redis, self.group_id, self.reason = redis, group_id, reason
        self.others = [_open(group_id), _open(group_id)]
        self.blocker: asyncio.Task | None = None
        self.queued: asyncio.Task | None = None
        self.hold = self.first_query_at = self.held_from = self.released_at = 0.0

    def held_seconds(self) -> float:
        """How long the blocker held the writer, from its sending to its
        reply."""
        assert self.blocker is not None and self.blocker.done(), "no blocker ran"
        return self.released_at - self.held_from

    async def execute_query(self, query: str, **params: Any) -> Any:
        self.first_query_at = self.first_query_at or time.perf_counter()
        if self.queued is not None or params.get("reason") != self.reason:
            return await super().execute_query(query, **params)
        await self._hold_the_writer()
        self.queued = asyncio.create_task(self.others[1].execute_query(query, **params))
        await asyncio.sleep(0.05)
        raise ConnectionError("the connection dropped with the write still queued")

    async def drain(self) -> None:
        """The blocker and the queued write finish; their connections close."""
        pending = [task for task in (self.blocker, self.queued) if task]
        await asyncio.gather(*pending, return_exceptions=True)
        for other in self.others:
            await other.close()

    async def _hold_the_writer(self) -> None:
        so_far = time.perf_counter() - self.first_query_at
        self.hold = max(_HOLD_SECONDS, _HOLD_FACTOR * so_far)
        self.held_from = time.perf_counter()
        self.blocker = asyncio.create_task(
            self.others[0].execute_query(_BLOCKER, hold_ms=int(self.hold * 1000))
        )
        self.blocker.add_done_callback(self._released)
        for _ in range(5000):
            info = str(
                await self.redis.execute_command(
                    "GRAPH.INFO", "RunningQueries", "WaitingQueries"
                )
            )
            if "holds the writer" in info and self.group_id in info:
                return
            assert not self.blocker.done(), "the blocker finished before it was seen"
            await asyncio.sleep(0.001)
        raise AssertionError("the blocker never ran")

    def _released(self, _: asyncio.Task) -> None:
        self.released_at = time.perf_counter()


def _open(group_id: str) -> AutoGPTFalkorDriver:
    return AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
        build_indices=False,
    )


@pytest.fixture
def live_pass(mocker, scope_graph) -> SimpleNamespace:
    """A dream pass on the live graph through the production gather, clamp
    and apply; only the chat store and the entity flag (on) are stubbed."""
    driver, scope = scope_graph
    mocker.patch.object(apply_mod, "_create_dream_session", return_value="s")
    mocker.patch.object(apply_mod, "_write_dream_summary_message")
    mocker.patch.object(demotions, "is_feature_enabled", return_value=True)
    state = SimpleNamespace(driver=driver, scope=scope, boundary=None)

    async def run(ops: DreamOperations, **boundary: Any) -> dict:
        state.boundary = _Boundary(driver, **boundary)
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
    "recalled, spared_by_the_demotion, changed_by_the_entity",
    [(False, False, []), (True, True, ["A"])],
    ids=["no-usage", "with-usage"],
)
async def test_case_b_a_user_retraction_through_an_entity_changes_no_more(
    live_pass,
    recalled: bool,
    spared_by_the_demotion: bool,
    changed_by_the_entity: list[str],
) -> None:
    """Two facts, a cap of one, stale proposals ``[A, B]`` and a
    ``user_signal`` invalidation of the entity only A hangs off. Without
    usage the direct write demotes A and the entity finds it gone; with A
    recalled the direct write spares it and the user's retraction demotes it
    through the entity. One edge changes either way (the first redo demoted
    B as well, 1 -> 2), and protection kept nothing live, so neither world
    counts a protected fact."""
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

    assert (_changed(stats), stats["protected_demotions"]) == (1, 0)
    snapshot = stats["snapshot"]
    assert snapshot.demotions[0].protected is spared_by_the_demotion
    assert snapshot.entity_invalidations[0].edges_touched == changed_by_the_entity
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


# FalkorDB answers a statement with at most RESULTSET_SIZE rows (10,000 by
# default); the hub below has more neighbours than that.
_HUB_SIZE = 12_000


async def _hub(
    driver: AutoGPTFalkorDriver, group_id: str, size: int, recalled: int
) -> None:
    """Entity ``hub`` with *size* live facts ``edge-0`` ... on their own
    leaves, the first *recalled* of them recalled a moment ago."""
    await driver.execute_query(
        """
        CREATE (h:Entity {uuid: 'hub', name: 'hub', group_id: $g})
        WITH h
        UNWIND range(0, $size - 1) AS i
        CREATE (h)-[:RELATES_TO {uuid: 'edge-' + toString(i), group_id: $g,
                                 name: 'knows', fact: 'hub fact',
                                 status: 'active', created_at: $created}]->
               (:Entity {uuid: 'leaf-' + toString(i), name: 'leaf',
                         group_id: $g})
        """,
        g=group_id,
        size=size,
        created=_ago(days=100),
    )
    await driver.execute_query(
        """
        MATCH (:Entity {uuid: 'hub'})-[r:RELATES_TO]->()
        WHERE toInteger(substring(r.uuid, 5)) < $recalled
        SET r.last_recalled_at = $now, r.recall_count = 1
        """,
        recalled=recalled,
        now=stamp_time(datetime.now(timezone.utc)),
    )


async def _status_counts(driver: AutoGPTFalkorDriver) -> dict[str, int]:
    found = await rows(
        driver,
        "MATCH ()-[r:RELATES_TO]->() RETURN r.status AS status, count(r) AS n",
    )
    return {row["status"]: row["n"] for row in found}


async def _stage(
    mocker, scope, ops: DreamOperations, known: set[str], boundary: _Boundary
):
    """The destructive stage alone, on the live graph behind *boundary*."""
    mocker.patch.object(demotions, "open_driver", return_value=boundary)
    mocker.patch.object(demotions, "is_feature_enabled", return_value=True)
    return await demotions.apply_demotions(scope, "p-stage", ops, known)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "recalled", [_HUB_SIZE, _HUB_SIZE // 2], ids=["all-protected", "half-and-half"]
)
async def test_a_hub_above_the_row_limit_is_accounted_in_full(
    mocker, scope_graph, recalled: int
) -> None:
    """Every one of the hub's 12,000 outcomes comes back, in one aggregate
    row, and the final read confirms every protected fact, in one row too.
    One row per neighbour would have stopped at the server's limit."""
    driver, scope = scope_graph
    await _hub(driver, scope.group_id, _HUB_SIZE, recalled)
    per_row = await rows(driver, "MATCH ()-[r:RELATES_TO]->() RETURN r.uuid AS uuid")
    assert len(per_row) < _HUB_SIZE, "the server's row limit truncates this"

    results = await _stage(
        mocker,
        scope,
        DreamOperations(
            entity_invalidations=[
                EntityInvalidation(entity_uuid="hub", reason="stale_fact")
            ]
        ),
        set(),
        _Boundary(driver),
    )

    [summary] = results.entity_invalidations
    changed = _HUB_SIZE - recalled
    assert (len(summary.edges_protected), len(summary.edges_touched)) == (
        recalled,
        changed,
    )
    assert len(set(summary.edges_protected) | set(summary.edges_touched)) == _HUB_SIZE
    assert (results.protected, results.accounting_complete) == (recalled, True)
    expected = {"active": recalled, "superseded": changed}
    assert await _status_counts(driver) == {k: v for k, v in expected.items() if v}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_fact_spared_then_overridden_through_a_big_hub_is_not_protected(
    mocker, scope_graph
) -> None:
    """Codex's case: a fact beyond the first 10,000 neighbours is spared by a
    direct stale demotion, then the user's retraction through its hub
    demotes all 12,000. None is kept live, and the pass says so."""
    driver, scope = scope_graph
    await _hub(driver, scope.group_id, _HUB_SIZE, _HUB_SIZE)
    target = f"edge-{_HUB_SIZE - 1}"

    results = await _stage(
        mocker,
        scope,
        DreamOperations(
            demotions=_stale(target),
            entity_invalidations=[
                EntityInvalidation(entity_uuid="hub", reason="user_signal")
            ],
        ),
        {target},
        _Boundary(driver),
    )

    assert results.demotions[0].protected is True
    assert (results.entity_edges, results.protected) == (_HUB_SIZE, 0)
    assert await _status_counts(driver) == {"superseded": _HUB_SIZE}


_OVERRIDES = pytest.mark.parametrize(
    "ops",
    [
        DreamOperations(
            demotions=[
                *_stale("A"),
                DreamDemotion(edge_uuid="A", reason="user_signal"),
            ]
        ),
        DreamOperations(
            demotions=_stale("A"),
            entity_invalidations=[
                EntityInvalidation(entity_uuid="hub", reason="user_signal")
            ],
        ),
    ],
    ids=["direct", "neighbour"],
)


@pytest.mark.integration
@pytest.mark.asyncio
@_OVERRIDES
async def test_a_lost_acknowledgement_is_indeterminate_never_protected(
    mocker, scope_graph, ops: DreamOperations
) -> None:
    """Codex's reproductions: A, recalled, is spared by a stale demotion; the
    user's retraction of it then commits, directly or through its entity,
    and its reply is lost. The write is indeterminate, not failed or empty;
    the final read finds A gone, so it is not counted as kept; and the count
    is provisional, since the pass cannot know the write committed."""
    driver, scope = scope_graph
    await _edge(driver, scope.group_id, "A", source="hub")
    assert await stamp_recalls(driver, ["A"], owner=_OWNER) == 1
    boundary = _Boundary(driver, lose_reason="user_signal")

    results = await _stage(mocker, scope, ops, {"A"}, boundary)

    assert boundary.lost_replies == 1
    assert results.demotions[0].protected is True
    assert (results.demoted, results.failed, results.indeterminate) == (0, 0, 1)
    assert (results.protected, results.accounting_complete) == (0, False)
    assert await _statuses(driver, ["A"]) == {"A": "superseded"}


@pytest.mark.integration
@pytest.mark.asyncio
@_OVERRIDES
async def test_a_write_still_queued_at_the_final_read_leaves_the_count_provisional(
    mocker, scope_graph, live_lock, ops: DreamOperations
) -> None:
    """Codex's in-flight reproduction: A, recalled, is spared by a stale
    demotion; the user's retraction of it reaches FalkorDB but waits behind
    another write, the pass's call raises, and the final read overtakes it.
    The read finds A live, as it then is, and counts it; the retraction
    lands after the stage returned. The count is provisional because a
    write's outcome is unknown, never complete. The blocker holds the writer
    for ``_HOLD_SECONDS`` or more whatever the server's speed, and the test
    fails loudly if the stage ever takes more than half as long."""
    driver, scope = scope_graph
    await _edge(driver, scope.group_id, "A", source="hub")
    assert await stamp_recalls(driver, ["A"], owner=_OWNER) == 1
    boundary = _Queued(driver, live_lock, scope.group_id, reason="user_signal")
    try:
        started = time.perf_counter()
        results = await _stage(mocker, scope, ops, {"A"}, boundary)
        stage_seconds = time.perf_counter() - started
        still_queued = boundary.queued is not None and not boundary.queued.done()
        at_return = await _statuses(driver, ["A"])
    finally:
        await boundary.drain()

    held = boundary.held_seconds()
    assert held >= 0.95 * boundary.hold, (
        f"the blocker held FalkorDB's writer for {held:.2f}s of its "
        f"{boundary.hold:.2f}s"
    )
    assert stage_seconds < held / 2, (
        f"the in-flight margin was exceeded: the stage took {stage_seconds:.2f}s, "
        f"more than half the {held:.2f}s the blocker held FalkorDB's writer"
    )
    assert still_queued, "the retraction was still queued when the stage returned"
    assert at_return == {"A": "active"}
    assert results.demotions[0].protected is True
    assert (results.demoted, results.failed, results.indeterminate) == (0, 0, 1)
    assert (results.protected, results.accounting_complete) == (1, False)
    assert await _statuses(driver, ["A"]) == {"A": "superseded"}


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fail_read, protected, complete",
    [(False, 1, False), (True, 2, False)],
    ids=["read-answers", "read-fails"],
)
async def test_a_full_pass_with_a_lost_acknowledgement_reports_the_truth(
    live_pass, fail_read: bool, protected: int, complete: bool
) -> None:
    """Codex's full-apply reproduction: A and B recalled, stale proposals
    ``[A, A, B]`` and the user's retraction of B, whose reply is lost after
    it committed. A alone is kept live, and the final read counts exactly
    that, provisionally, since the pass cannot know what the retraction did;
    if the read fails too, the count falls back to spared minus acknowledged
    changes. Neither is ever reported as complete."""
    driver, scope = live_pass.driver, live_pass.scope
    for uuid in ["A", "B", *(f"filler-{i}" for i in range(98))]:
        await _edge(driver, scope.group_id, uuid)
    assert await stamp_recalls(driver, ["A", "B"], owner=_OWNER) == 2

    stats = await live_pass.run(
        DreamOperations(
            demotions=[
                *_stale("A", "A", "B"),
                DreamDemotion(edge_uuid="B", reason="user_signal"),
            ]
        ),
        lose_reason="user_signal",
        fail_read=fail_read,
    )

    assert (
        stats["demotion_count"],
        stats["demotion_failed_count"],
        stats["indeterminate_demotion_writes"],
        stats["protected_demotions"],
        stats["demotion_accounting_complete"],
    ) == (0, 0, 1, protected, complete)
    assert await _statuses(driver, ["A", "B"]) == {"A": "active", "B": "superseded"}
