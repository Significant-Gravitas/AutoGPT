"""Reinier's scenario (#13776 review): does an absence (a two-week holiday,
rotating between projects) make memories look unused and get them demoted?

For a fixed set of demotions and entity invalidations the sanitizer proposes,
run the real clamp and apply twice: once with the pass's recall stamps (and
stamps read again at apply time that may be newer still), once with no recall
history at all. With usage data present the pass never demotes more, for a
user recalled yesterday, a user away for 20 days, a user never recalled, and
generated histories of every shape. The stamps can only take demotions away.
"""

import random
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from backend.copilot.graphiti.recall_stamp import RecallStamp, stamp_time
from backend.copilot.graphiti.scope import MemoryScope

from . import apply as apply_mod
from . import recall_guard
from .clamp import clamp_pass_operations
from .fetch import DreamInput, FactRow
from .schemas import DreamDemotion, DreamOperations, EntityInvalidation

_SCOPE = MemoryScope.for_user("u-reinier")
_CASES_PER_SCENARIO = 60
_REASONS = [
    "stale_fact",
    "stale_fact",
    "user_signal",
    "contradicted_by:{other}",
    "contradicted_by:{target}",
    "contradicted_by:ghost",
    "web_contradicted:https://example.test",
    "entity_invalidated:hub",
    "no longer seems relevant",
]


def _days_since_last_recall(rng: random.Random, scenario: str) -> float | None:
    """How long before the gather a fact was last recalled; ``None``: never."""
    if scenario == "never":
        return None
    if rng.random() < 0.3:
        return None
    if scenario == "recalled_yesterday":
        return rng.uniform(0.5, 1.5)
    if scenario == "away_20_days":
        return 20 + rng.uniform(0, 40)
    return rng.uniform(0, 120)


class _Case(BaseModel):
    """One generated pass: its facts as gathered, the stamps apply reads
    again, what the sanitizer proposed, and each invalidated entity's
    neighbours."""

    facts: list[FactRow]
    fresh: list[RecallStamp]
    ops: DreamOperations
    neighbours: dict[str, list[str]]


def _case(rng: random.Random, scenario: str) -> _Case:
    now = datetime.now(timezone.utc)
    facts: list[FactRow] = []
    fresh: list[RecallStamp] = []
    for i in range(rng.randint(1, 120)):
        days = _days_since_last_recall(rng, scenario)
        last = stamp_time(now - timedelta(days=days)) if days is not None else None
        facts.append(_fact(f"f{i}", last))
        # A recall after the gather only ever makes the last one more recent.
        if scenario != "never" and rng.random() < 0.2:
            last = stamp_time(now - timedelta(hours=rng.uniform(0, 6)))
        if last is not None:
            fresh.append(
                RecallStamp(uuid=f"f{i}", recall_count=1, last_recalled_at=last)
            )
    uuids = [f.uuid for f in facts]
    demotions = [
        DreamDemotion(
            edge_uuid=target,
            reason=rng.choice(_REASONS).format(other=rng.choice(uuids), target=target),
        )
        for target in rng.choices(uuids + ["hallucinated"], k=rng.randint(0, 15))
    ]
    neighbours = {
        f"hub{j}": rng.sample(uuids, k=min(len(uuids), rng.randint(0, 6)))
        for j in range(rng.randint(0, 2))
    }
    invalidations = [
        EntityInvalidation(entity_uuid=hub, reason=rng.choice(_REASONS[:3]))
        for hub in neighbours
    ]
    return _Case(
        facts=facts,
        fresh=fresh,
        ops=DreamOperations(
            demotions=demotions,
            entity_invalidations=invalidations,
            summary_for_user="x",
        ),
        neighbours=neighbours,
    )


def _fact(uuid: str, last_recalled_at: str | None) -> FactRow:
    return FactRow(
        uuid=uuid,
        source="A",
        target="B",
        name="r",
        fact=f"fact {uuid}",
        scope="real:global",
        confidence=0.5,
        status="active",
        created_at="2026-01-01T00:00:00+00:00",
        recall_count=1 if last_recalled_at else None,
        last_recalled_at=last_recalled_at,
    )


def _without_history(facts: list[FactRow]) -> list[FactRow]:
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


@pytest.fixture
def graph(mocker) -> SimpleNamespace:
    """apply's graph writes, counting every fact demoted; the guard's reads
    answer from the case in play."""
    state = SimpleNamespace(stamps=[], neighbours={})

    async def read(driver, group_id, uuids):
        return [s for s in state.stamps if s.uuid in uuids]

    async def read_neighbours(driver, group_id, entity_uuid):
        by_uuid = {s.uuid: s for s in state.stamps}
        return [
            by_uuid.get(uuid, RecallStamp(uuid=uuid))
            for uuid in state.neighbours[entity_uuid]
        ]

    async def invalidate(driver, *, group_id, entity_uuid, reason, skip=()):
        return [u for u in state.neighbours[entity_uuid] if u not in skip]

    driver = MagicMock()
    driver.close = AsyncMock()
    mocker.patch.object(apply_mod, "open_driver", return_value=driver)
    mocker.patch.object(
        apply_mod,
        "mark_edges_superseded",
        AsyncMock(side_effect=lambda driver, uuids, **kw: (list(uuids), [])),
    )
    mocker.patch.object(
        apply_mod, "invalidate_entity_direct_neighbors", side_effect=invalidate
    )
    mocker.patch.object(recall_guard, "read_recall_stamps", side_effect=read)
    mocker.patch.object(
        recall_guard, "read_neighbour_stamps", side_effect=read_neighbours
    )
    mocker.patch.object(apply_mod, "is_feature_enabled", AsyncMock(return_value=True))
    database = MagicMock()
    database.create_chat_session = AsyncMock()
    database.update_chat_session_title = AsyncMock()
    database.add_chat_message = AsyncMock()
    mocker.patch("backend.data.db_accessors.chat_db", return_value=database)
    mocker.patch(
        "backend.api.features.orgs.db.get_user_default_team",
        AsyncMock(return_value=(None, None)),
    )
    return state


async def _demoted(
    graph: SimpleNamespace, case: _Case, facts: list[FactRow], fresh: list[RecallStamp]
) -> int:
    """How many facts the pass demotes, through the real clamp and apply."""
    graph.stamps, graph.neighbours = fresh, case.neighbours
    now = datetime.now(timezone.utc)
    bundle = DreamInput(
        user_id=_SCOPE.owner_user_id,
        group_id=_SCOPE.group_id,
        window_start=now,
        window_end=now,
        facts=facts,
        known_fact_uuids={f.uuid for f in facts},
    )
    clamped = clamp_pass_operations(case.ops, bundle)
    stats = await apply_mod.apply_operations(
        _SCOPE,
        "p-reinier",
        clamped.ops,
        known_fact_uuids=bundle.known_fact_uuids,
        protected_demotions=clamped.protected_demotions,
    )
    return int(stats["demotion_count"]) + int(stats["entity_invalidation_count"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario", ["recalled_yesterday", "away_20_days", "never", "any_history"]
)
async def test_usage_data_never_raises_the_number_of_demotions(
    graph, scenario: str
) -> None:
    rng = random.Random(f"reinier-{scenario}")
    for _ in range(_CASES_PER_SCENARIO):
        case = _case(rng, scenario)

        with_usage = await _demoted(graph, case, case.facts, case.fresh)
        without = await _demoted(graph, case, _without_history(case.facts), [])

        assert with_usage <= without, case
        if scenario == "never":
            assert with_usage == without, case


@pytest.mark.asyncio
async def test_twenty_days_away_leaves_relied_on_memories_alone(graph) -> None:
    """The holiday itself: every fact the user relied on was last recalled 20
    days before the pass. Under the default 30-day window each is still
    protected, so the proposed staleness demotions all drop, where the 8-day
    window of #13776 would have let every one through."""
    now = datetime.now(timezone.utc)
    last = stamp_time(now - timedelta(days=20))
    facts = [_fact(f"f{i}", last) for i in range(100)]
    fresh = [
        RecallStamp(uuid=f.uuid, recall_count=1, last_recalled_at=last) for f in facts
    ]
    case = _Case(
        facts=facts,
        fresh=fresh,
        ops=DreamOperations(
            demotions=[
                DreamDemotion(edge_uuid=f"f{i}", reason="stale_fact") for i in range(5)
            ],
            summary_for_user="x",
        ),
        neighbours={},
    )

    assert await _demoted(graph, case, case.facts, case.fresh) == 0
    assert await _demoted(graph, case, _without_history(case.facts), []) == 5
