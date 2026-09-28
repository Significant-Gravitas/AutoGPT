"""Usage never raises the number of facts a pass demotes, and no fact recalled
within the window when its write runs is demoted without a valid override: a
stateful proof, answering Reinier's review of #13776 (a holiday, a rotation
between projects).

A simulated graph stands in for FalkorDB behind the destructive stage's two
guarded writers: edges with liveness, recall stamps and the two entities they
join, and the rule each guarded statement enforces (a live edge is changed
unless the write's protection spares it; anything else is left as it is).
Operations run in order on that state: an edge demoted once is not demoted
again, an expired neighbour is not changed, a target a forget took between
gather and apply fails, and duplicates, direct/entity overlaps and overrides
are all generated, as are writes whose reply never reaches the pass (one
that committed, one that never arrived, and one still queued on the server,
landing between later statements or after the pass returned) and a final
liveness read that fails. The real clamp and apply run each generated case
twice: with the recall stamps, and in the world before them (no stamps
anywhere, recalls stamping nothing). The pass's account is held to what it
can know: acknowledged changes, the writes whose outcome it cannot know, and
the facts protection kept live as its final read saw them. That count is
complete only when the read answered and no write's outcome is unknown, and
then it is the truth once every queued write has landed; otherwise it is
provisional, and never below that truth. What the statements themselves do
is held to the same rule on FalkorDB by
``graphiti/recall_guard_integration_test.py``.
"""

import random
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from functools import partial
from types import SimpleNamespace
from typing import Literal, TypeVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from backend.copilot.graphiti.guarded_writes import NeighbourWrites, WriteOutcome
from backend.copilot.graphiti.recall_stamp import RecallProtection, stamp_time
from backend.copilot.graphiti.scope import MemoryScope

from . import apply as apply_mod
from . import demotions as demotions_mod
from .clamp import clamp_pass_operations
from .fetch import DreamInput, FactRow
from .schemas import DreamDemotion, DreamOperations, EntityInvalidation

_SCOPE = MemoryScope.for_user("u-stateful")
_WINDOW = timedelta(days=30)
_CASES = 60
_ENTITIES = [f"n{i}" for i in range(8)]
# Days since the last recall, per history; none within a day of the window's
# edge, so the oracle's clock and the stage's cannot disagree.
_HISTORIES = {
    "yesterday": (0.5, 1.5),
    "away_20_days": (19.0, 21.0),
    "away_40_days": (39.0, 41.0),
    "away_90_days": (85.0, 95.0),
}
_REASONS = [
    "stale_fact",
    "stale_fact",
    "user_signal",
    "contradicted_by:{other}",
    "contradicted_by:{target}",
    "contradicted_by:ghost",
    "web_contradicted:https://example.test",
    "entity_invalidated:n1",
    "no longer seems relevant",
]
# The faults each scenario draws its lost replies from.
_FAULTS: dict[str, tuple[str, ...]] = {
    "lost_acknowledgements": ("after_commit",),
    "never_delivered": ("never_delivered",),
    "in_flight": ("in_flight",),
    "failed_read": (),
    "any": ("after_commit", "never_delivered", "in_flight"),
}


class Edge(BaseModel):
    uuid: str
    ends: tuple[str, str]
    live: bool = True
    last_recalled_at: datetime | None = None


class Write(BaseModel):
    """One live edge a guarded write reached, as the graph stood then, and
    whether the write's reply reached the pass."""

    edge: str
    reason: str
    changed: bool
    last_recalled_at: datetime | None
    acknowledged: bool = True


class Fault(BaseModel):
    """Why one statement's reply never reaches the pass: it committed and the
    reply was lost, it never arrived, or it was still queued on the server
    when the pass went on. A queued write lands right before statement
    ``lands_before`` runs or, when that is None, after the pass returned."""

    kind: Literal["after_commit", "never_delivered", "in_flight"]
    lands_before: int | None = None


_Reply = TypeVar("_Reply", WriteOutcome, NeighbourWrites)


class Graph:
    """The simulated graph behind the stage's two writers."""

    def __init__(self, case: "Case", usage: bool) -> None:
        self.edges = {e.uuid: e.model_copy() for e in case.edges}
        self.recalls = case.recalls
        self.faults = case.faults
        self.read_fails = case.read_fails
        self.usage = usage
        self.statements = 0
        self.writes: list[Write] = []
        self.queued: list[tuple[int | None, Callable[[bool], object]]] = []
        self.live_at_read: set[str] | None = None

    async def supersede(
        self,
        driver: object,
        uuids: list[str],
        *,
        reason: str,
        new_status: str,
        group_id: str,
        protection: RecallProtection,
        user_id: str | None = None,
    ) -> list[WriteOutcome]:
        return [
            self._statement(
                partial(self._supersede_one, uuid, reason, protection),
                WriteOutcome.UNKNOWN,
            )
            for uuid in uuids
        ]

    async def invalidate(
        self,
        driver: object,
        *,
        group_id: str,
        entity_uuid: str,
        reason: str,
        protection: RecallProtection,
    ) -> NeighbourWrites:
        return self._statement(
            partial(self._invalidate, entity_uuid, reason, protection),
            NeighbourWrites(unknown=True),
        )

    async def live_uuids(
        self, driver: object, group_id: str, uuids: list[str]
    ) -> set[str]:
        """The stage's final liveness read. Like ``GRAPH.RO_QUERY`` it does
        not wait for writes still queued."""
        if self.read_fails:
            raise TimeoutError("the liveness read's reply was lost")
        self.live_at_read = {uuid for uuid in uuids if self.edges[uuid].live}
        return self.live_at_read

    def land_in_flight(self) -> None:
        """The pass has returned: every write still queued lands."""
        for _, run in self.queued:
            run(False)
        self.queued = []

    def changed(self) -> set[str]:
        """Every fact a write changed, acknowledged or not."""
        return {w.edge for w in self.writes if w.changed}

    def acknowledged(self, *, changed: bool) -> set[str]:
        """The facts an acknowledged write changed, or spared."""
        return {w.edge for w in self.writes if w.acknowledged and w.changed is changed}

    def kept_live(self) -> int:
        """The distinct facts an acknowledged write spared that are live
        now."""
        return len(
            {edge for edge in self.acknowledged(changed=False) if self.edges[edge].live}
        )

    def indeterminate(self) -> int:
        """The statements that ran with a fault: no reply reached the pass."""
        return len([k for k in self.faults if k < self.statements])

    def _statement(self, run: Callable[[bool], _Reply], unknown: _Reply) -> _Reply:
        """One statement as the pass sees it. The queued writes due by now
        land and the recalls that complete right before it stamp; then it
        runs, unless its fault drops it or leaves it queued, and its reply
        reaches the pass only when it has no fault."""
        k = self.statements
        self.statements += 1
        self._land(before=k)
        self._recall(k)
        fault = self.faults.get(k)
        if fault is None:
            return run(True)
        if fault.kind == "after_commit":
            run(False)
        elif fault.kind == "in_flight":
            self.queued.append((fault.lands_before, run))
        return unknown

    def _land(self, before: int) -> None:
        """The queued writes due to land before statement *before* land."""
        due = [run for at, run in self.queued if at is not None and at <= before]
        self.queued = [
            (at, run) for at, run in self.queued if at is None or at > before
        ]
        for run in due:
            run(False)

    def _recall(self, k: int) -> None:
        """The recalls that complete right before statement *k* runs."""
        for uuid in self.recalls.get(k, []):
            edge = self.edges.get(uuid)
            if self.usage and edge is not None and edge.live:
                edge.last_recalled_at = datetime.now(timezone.utc)

    def _supersede_one(
        self,
        uuid: str,
        reason: str,
        protection: RecallProtection,
        acknowledged: bool,
    ) -> WriteOutcome:
        edge = self.edges.get(uuid)
        if edge is None or not edge.live:
            return WriteOutcome.UNMATCHED
        if self._write(edge, reason, protection, acknowledged):
            return WriteOutcome.CHANGED
        return WriteOutcome.SPARED

    def _invalidate(
        self,
        entity_uuid: str,
        reason: str,
        protection: RecallProtection,
        acknowledged: bool,
    ) -> NeighbourWrites:
        writes = NeighbourWrites()
        for edge in self.edges.values():
            if entity_uuid in edge.ends and edge.live:
                bucket = (
                    writes.changed
                    if self._write(edge, reason, protection, acknowledged)
                    else writes.spared
                )
                bucket.append(edge.uuid)
        return writes

    def _write(
        self,
        edge: Edge,
        reason: str,
        protection: RecallProtection,
        acknowledged: bool,
    ) -> bool:
        """The guarded statement's rule on one live edge; whether it changed."""
        spared = (
            protection.recalled_since is not None
            and edge.last_recalled_at is not None
            and stamp_time(edge.last_recalled_at) >= protection.recalled_since
            and not (
                protection.override
                and (protection.cited is None or edge.uuid != protection.cited)
            )
        )
        self.writes.append(
            Write(
                edge=edge.uuid,
                reason=reason,
                changed=not spared,
                last_recalled_at=edge.last_recalled_at,
                acknowledged=acknowledged,
            )
        )
        edge.live = edge.live and spared
        return not spared


class Case(BaseModel):
    """One generated pass: the graph as gathered, what changed between
    gather and apply, the recalls that land during apply, the statements
    whose reply never reaches the pass, whether the final read fails, and
    the fixed proposals."""

    edges: list[Edge]
    forgotten_after_gather: list[str]
    recalled_after_gather: list[str]
    recalls: dict[int, list[str]]
    ops: DreamOperations
    faults: dict[int, Fault] = {}
    read_fails: bool = False

    def gathered(self) -> list[Edge]:
        return [e for e in self.edges if e.live]


@pytest.fixture
def world(mocker) -> SimpleNamespace:
    """The stage's writers bound to whichever simulated graph is current, and
    apply's other boundaries (chat store, flag on)."""
    holder = SimpleNamespace(graph=None)

    async def supersede(*args, **kwargs):
        return await holder.graph.supersede(*args, **kwargs)

    async def invalidate(*args, **kwargs):
        return await holder.graph.invalidate(*args, **kwargs)

    async def live_uuids(*args, **kwargs):
        return await holder.graph.live_uuids(*args, **kwargs)

    driver = MagicMock()
    driver.close = AsyncMock()
    mocker.patch.object(demotions_mod, "open_driver", return_value=driver)
    mocker.patch.object(
        demotions_mod, "supersede_unless_recalled", AsyncMock(side_effect=supersede)
    )
    mocker.patch.object(
        demotions_mod,
        "invalidate_entity_direct_neighbors",
        AsyncMock(side_effect=invalidate),
    )
    mocker.patch.object(
        demotions_mod, "live_fact_uuids", AsyncMock(side_effect=live_uuids)
    )
    mocker.patch.object(
        demotions_mod, "is_feature_enabled", AsyncMock(return_value=True)
    )
    database = MagicMock()
    database.create_chat_session = AsyncMock()
    database.update_chat_session_title = AsyncMock()
    database.add_chat_message = AsyncMock()
    mocker.patch("backend.data.db_accessors.chat_db", return_value=database)
    mocker.patch(
        "backend.api.features.orgs.db.get_user_default_team",
        AsyncMock(return_value=(None, None)),
    )
    return holder


def _stamp(rng: random.Random, history: str, now: datetime) -> datetime | None:
    if history == "any":
        history = rng.choice(["never", *_HISTORIES])
    if history not in _HISTORIES or rng.random() < 0.3:
        return None
    low, high = _HISTORIES[history]
    return now - timedelta(days=rng.uniform(low, high))


def _case(rng: random.Random, scenario: str) -> Case:
    now = datetime.now(timezone.utc)
    lossy = scenario in _FAULTS
    kinds = _FAULTS.get(scenario, ())
    history = scenario if scenario in (*_HISTORIES, "any") else "never"
    history = "any" if lossy else history
    edges = [
        Edge(
            uuid=f"f{i}",
            ends=(rng.choice(_ENTITIES), rng.choice(_ENTITIES)),
            live=rng.random() > 0.1,
            last_recalled_at=_stamp(rng, history, now),
        )
        for i in range(rng.randint(1, 40))
    ]
    gathered = [e.uuid for e in edges if e.live]
    late = scenario in ("recalled_after_gather", "any")
    during = scenario in ("recalled_during_apply", "any")
    ops = _proposals(rng, edges, gathered)
    if lossy and gathered and rng.random() < 0.7:
        _spare_then_override(rng, edges, gathered, ops, now)
    return Case(
        edges=edges,
        forgotten_after_gather=[u for u in gathered if rng.random() < 0.1],
        recalled_after_gather=[u for u in gathered if late and rng.random() < 0.4],
        recalls={
            k: rng.sample([e.uuid for e in edges], k=min(len(edges), 3))
            for k in range(30)
            if during and rng.random() < 0.3
        },
        ops=ops,
        faults={
            k: _fault(rng, k, kinds) for k in range(30) if kinds and rng.random() < 0.3
        },
        read_fails=scenario == "failed_read"
        or (scenario == "any" and rng.random() < 0.2),
    )


def _fault(rng: random.Random, k: int, kinds: tuple[str, ...]) -> Fault:
    """A fault for statement *k*; a queued write lands before the next
    statement, a few later, or after the pass returned."""
    kind = rng.choice(kinds)
    lands_before = rng.choice([None, None, k + 1, k + rng.randint(2, 6)])
    return Fault.model_validate(
        {"kind": kind, "lands_before": lands_before if kind == "in_flight" else None}
    )


def _spare_then_override(
    rng: random.Random,
    edges: list[Edge],
    gathered: list[str],
    ops: DreamOperations,
    now: datetime,
) -> None:
    """Codex's lost-acknowledgement and in-flight shape: a recently recalled
    fact the first direct write spares, then the user's retraction through
    one of its entities, whose write may commit with its reply lost, never
    arrive, or still be queued when the pass reads."""
    chosen = rng.choice(gathered)
    target = next(e for e in edges if e.uuid == chosen)
    target.last_recalled_at = now - timedelta(days=1)
    ops.demotions.insert(0, DreamDemotion(edge_uuid=target.uuid, reason="stale_fact"))
    ops.entity_invalidations.insert(
        0, EntityInvalidation(entity_uuid=target.ends[0], reason="user_signal")
    )


def _proposals(
    rng: random.Random, edges: list[Edge], gathered: list[str]
) -> DreamOperations:
    others = gathered or ["ghost"]
    targets = gathered + ["hallucinated"]
    demotions = [
        DreamDemotion(
            edge_uuid=target,
            reason=rng.choice(_REASONS).format(other=rng.choice(others), target=target),
            new_status=rng.choice(["superseded", "contradicted"]),
        )
        for target in rng.choices(targets, k=rng.randint(0, 12))
    ]
    invalidations = []
    for entity in rng.sample(_ENTITIES, k=rng.randint(0, 3)):
        attached = [e.uuid for e in edges if entity in e.ends] or ["ghost"]
        reason = rng.choice(_REASONS[:6]).format(
            other=rng.choice(others), target=rng.choice(attached)
        )
        invalidations.append(EntityInvalidation(entity_uuid=entity, reason=reason))
    return DreamOperations(
        demotions=demotions, entity_invalidations=invalidations, summary_for_user="x"
    )


async def _run(world: SimpleNamespace, case: Case, usage: bool) -> tuple[Graph, dict]:
    """The real clamp and apply on *case*, with or without usage data; then
    the writes still queued when the pass returned land."""
    graph = Graph(case, usage)
    now = datetime.now(timezone.utc)
    for edge in graph.edges.values():
        if not usage:
            edge.last_recalled_at = None
        elif edge.uuid in case.recalled_after_gather:
            edge.last_recalled_at = now
        if edge.uuid in case.forgotten_after_gather:
            edge.live = False
    world.graph = graph
    bundle = DreamInput(
        user_id=_SCOPE.owner_user_id,
        group_id=_SCOPE.group_id,
        window_start=now,
        window_end=now,
        facts=[_fact(edge, usage) for edge in case.gathered()],
        known_fact_uuids={edge.uuid for edge in case.gathered()},
    )
    stats = await apply_mod.apply_operations(
        _SCOPE,
        "p-stateful",
        clamp_pass_operations(case.ops, bundle),
        known_fact_uuids=bundle.known_fact_uuids,
    )
    graph.land_in_flight()
    return graph, stats


def _fact(edge: Edge, usage: bool) -> FactRow:
    last = edge.last_recalled_at if usage else None
    return FactRow(
        uuid=edge.uuid,
        source=edge.ends[0],
        target=edge.ends[1],
        name="r",
        fact=f"fact {edge.uuid}",
        scope="real:global",
        confidence=0.5,
        status="active",
        created_at="2026-01-01T00:00:00+00:00",
        recall_count=1 if last else None,
        last_recalled_at=stamp_time(last) if last else None,
    )


def _override_is_valid(reason: str, edge: str, known: set[str]) -> bool:
    """The override vocabulary, written out again for the oracle."""
    if reason == "user_signal":
        return True
    cited = reason.removeprefix("contradicted_by:").strip()
    return reason.startswith("contradicted_by:") and cited in known and cited != edge


def _check_every_write(graph: Graph, known: set[str]) -> None:
    """Each live edge a write reached was changed exactly when it was not
    recalled within the window, or the write's override was valid."""
    cutoff = datetime.now(timezone.utc) - _WINDOW
    for write in graph.writes:
        recent = write.last_recalled_at is not None and write.last_recalled_at >= cutoff
        protected = recent and not _override_is_valid(write.reason, write.edge, known)
        assert write.changed is not protected, write


def _check_the_account(graph: Graph, stats: dict, case: Case) -> None:
    """The pass reports what it can know: acknowledged changes; the writes
    whose outcome it cannot know; and the facts an acknowledged write spared
    that its final read found live, or, when that read failed, spared minus
    acknowledged changes. The count is complete only when the read answered
    and no write's outcome is unknown, and then it is the truth once every
    queued write has landed; a provisional count is never below it."""
    changed = graph.acknowledged(changed=True)
    reported = int(stats["demotion_count"]) + int(stats["entity_invalidation_count"])
    assert reported == len(changed), case
    assert stats["indeterminate_demotion_writes"] == graph.indeterminate(), case
    spared = graph.acknowledged(changed=False)
    read = graph.live_at_read
    complete = graph.indeterminate() == 0 and (read is not None or not spared)
    assert stats["demotion_accounting_complete"] is complete, case
    counted = spared & read if read is not None else spared - changed
    assert stats["protected_demotions"] == len(counted), case
    assert stats["protected_demotions"] >= graph.kept_live(), case
    if complete:
        assert stats["protected_demotions"] == graph.kept_live(), case


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "never",
        "yesterday",
        "away_20_days",
        "away_40_days",
        "away_90_days",
        "recalled_after_gather",
        "recalled_during_apply",
        "lost_acknowledgements",
        "never_delivered",
        "in_flight",
        "failed_read",
        "any",
    ],
)
async def test_usage_never_demotes_more_and_protects_at_every_write(
    world, scenario: str
) -> None:
    rng = random.Random(f"stateful-{scenario}")
    for _ in range(_CASES):
        case = _case(rng, scenario)
        known = {edge.uuid for edge in case.gathered()}

        with_usage, stats = await _run(world, case, usage=True)
        without, _ = await _run(world, case, usage=False)

        assert with_usage.changed() <= without.changed(), case
        assert len(with_usage.changed()) <= len(without.changed()), case
        if scenario in ("never", "away_40_days", "away_90_days"):
            assert with_usage.changed() == without.changed(), case
        _check_every_write(with_usage, known)
        _check_the_account(with_usage, stats, case)


@pytest.mark.asyncio
async def test_twenty_days_away_leaves_relied_on_memories_alone(world) -> None:
    """The holiday itself: every fact the user relied on was last recalled 20
    days before the pass. Under the default 30-day window the writes spare
    each one; #13776's 8 days would have protected none."""
    now = datetime.now(timezone.utc)
    edges = [
        Edge(uuid=f"f{i}", ends=("n0", "n1"), last_recalled_at=now - timedelta(days=20))
        for i in range(100)
    ]
    case = Case(
        edges=edges,
        forgotten_after_gather=[],
        recalled_after_gather=[],
        recalls={},
        ops=DreamOperations(
            demotions=[
                DreamDemotion(edge_uuid=f"f{i}", reason="stale_fact") for i in range(5)
            ],
            summary_for_user="x",
        ),
    )

    with_usage, stats = await _run(world, case, usage=True)
    without, _ = await _run(world, case, usage=False)

    assert (len(with_usage.changed()), len(without.changed())) == (0, 5)
    assert stats["protected_demotions"] == 5
