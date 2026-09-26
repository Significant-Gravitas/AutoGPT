"""Unit tests for ``recall_ingest_plan``: what the forget repair decides
after ``add_episode``, against a mock driver.

``recall_ingest_test.py`` pins how the repair carries a plan out; the live
runs, through the production worker, are ``recall_ingest_integration_test.py``,
``recall_repair_integration_test.py`` and ``recall_inflight_integration_test.py``.
"""

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import AddEpisodeResults
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

from .recall import FORGOTTEN_FACT, forgotten_fact_predicate
from .recall_ingest_plan import IngestRun, Plan, Reapply, make_plan, snapshot_forgotten
from .recall_restore import FIELDS, ForgottenEdge
from .recall_stash import ForgetRecord

_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
_LONG_AGO = "2026-01-01T00:00:00+00:00"
_SENTENCE = "Alice works on Atlas"
# The episode was said 30s before its ingestion began at _NOW.
_RUN = IngestRun(
    group_id="user_test",
    started_at=_NOW,
    reference_time=_NOW - timedelta(seconds=30),
    previous=[],
)
_MID_EPISODE = (_NOW + timedelta(seconds=10)).isoformat()
_AFTER_SAID = (_NOW - timedelta(seconds=20)).isoformat()


def _record(when: str = _LONG_AGO, **update: Any) -> ForgetRecord:
    return ForgetRecord(
        uuid="f1",
        forgotten_at=when,
        expired_at=when,
        fact_redacted=_SENTENCE,
        name_redacted="MemoryFact",
        source="alice",
        target="atlas",
        source_name="Alice",
        target_name="Atlas",
        stashed_at=datetime.now(timezone.utc).isoformat(),
        **update,
    )


def _state(when: str = _LONG_AGO, **fields: Any) -> ForgottenEdge:
    """The edge as a finished forget left it, with ``fields`` changed."""
    base = {field: None for field in FIELDS}
    base |= _record(when).model_dump(include=set(base))
    base |= {"episodes": ["ep-old"], "provenance": "session:s1#msg:1"}
    return ForgottenEdge(
        uuid="f1", source="alice", target="atlas", fields=base | fields
    )


def _edge(uuid: str, fact: str = FORGOTTEN_FACT, source: str = "alice") -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        group_id="user_test",
        source_node_uuid=source,
        target_node_uuid="atlas",
        created_at=_NOW,
        name="MemoryFact",
        fact=fact,
        episodes=["ep-new"],
    )


def _result(*edges: EntityEdge, cites: list[str]) -> AddEpisodeResults:
    episode = EpisodicNode(
        uuid="ep-new",
        name="conversation_s2",
        group_id="user_test",
        source=EpisodeType.message,
        source_description="User message in session s2",
        content=_SENTENCE,
        valid_at=_NOW,
        entity_edges=cites,
    )
    nodes = [
        EntityNode(uuid=uuid, name=name, group_id="user_test")
        for uuid, name in (("alice", "Alice"), ("atlas", "Atlas"), ("a2", "alice"))
    ]
    return AddEpisodeResults(
        episode=episode,
        episodic_edges=[],
        nodes=nodes,
        edges=list(edges),
        communities=[],
        community_edges=[],
    )


def _driver(*states: ForgottenEdge) -> AsyncMock:
    rows = [
        {"uuid": s.uuid, "source": s.source, "target": s.target, **s.fields}
        for s in states
    ]
    driver = AsyncMock()
    driver.execute_query.return_value = (rows, [], None)
    return driver


async def _plan(
    before: list[ForgottenEdge], stashed: list[ForgetRecord], result, *states
) -> Plan:
    return await make_plan(
        _driver(*states),
        _RUN,
        {edge.uuid: edge for edge in before},
        {record.uuid: record for record in stashed},
        result,
    )


class TestMakePlan:
    @pytest.mark.asyncio
    async def test_a_forget_that_landed_mid_episode_is_applied_again(self) -> None:
        state = _state(forgotten_at=None, fact=_SENTENCE)  # graphiti's older copy
        plan = await _plan([], [_record(_MID_EPISODE)], _result(cites=[]), state)

        assert plan.reapply == [Reapply(uuid="f1", hard=False, reason="user_signal")]
        assert plan.restores == [], "applied again as a whole instead"

    @pytest.mark.asyncio
    async def test_a_hard_forget_whose_edge_stayed_gone_has_its_ends_scrubbed(
        self,
    ) -> None:
        plan = await _plan([], [_record(_MID_EPISODE, hard=True)], _result(cites=[]))

        assert (plan.reapply, plan.scrub) == ([], ["alice", "atlas"])

    @pytest.mark.asyncio
    async def test_a_fact_said_again_after_its_forget_is_merged_out(self) -> None:
        before = _state()
        merged = _state(episodes=["ep-old", "ep-new"], forgotten_at=None)
        plan = await _plan([before], [], _result(_edge("f1"), cites=["f1"]), merged)

        [spec] = plan.merged
        assert spec.dropped == ["ep-new"]
        assert plan.unlink == ["f1"] and plan.restores == [spec]
        assert plan.covered == []

    @pytest.mark.asyncio
    async def test_an_episode_said_before_the_forget_stays_its_source(self) -> None:
        """Queued before the forget, ingested after: graphiti merged it into
        the forgotten edge, where it stays, hidden with it."""
        merged = _state(_AFTER_SAID, episodes=["ep-old", "ep-new"])
        stashed = [_record(_AFTER_SAID)]
        plan = await _plan([], stashed, _result(_edge("f1"), cites=["f1"]), merged)

        assert [spec.uuid for spec in plan.covered] == ["f1"]
        assert (plan.unlink, plan.merged) == ([], [])
        assert plan.restores == [], "the forget itself is intact"

    @pytest.mark.asyncio
    async def test_an_episode_that_only_contradicts_it_stops_citing_it(self) -> None:
        stamped = _state(invalid_at=_NOW.isoformat())
        plan = await _plan([_state()], [], _result(_edge("f1"), cites=["f1"]), stamped)

        assert (plan.unlink, plan.merged, plan.covered) == (["f1"], [], [])
        [spec] = plan.restores
        assert spec.exact["invalid_at"] is None, "graphiti's stamp is undone"

    @pytest.mark.asyncio
    async def test_an_intact_forget_the_episode_did_not_touch_is_left(self) -> None:
        plan = await _plan([], [_record()], _result(cites=[]), _state())

        assert plan.model_dump(exclude={"forgotten"}) == Plan().model_dump(
            exclude={"forgotten"}
        )

    @pytest.mark.asyncio
    async def test_a_repair_left_undone_is_done_by_the_next_ingestion(self) -> None:
        """A restore that failed left its record; this episode is unrelated."""
        undone = _state(
            forgotten_at=None, fact_redacted=None, episodes=["ep-old", "ep-x"]
        )
        stashed = [_record(dropped_episodes=["ep-x"])]
        plan = await _plan([], stashed, _result(cites=[]), undone)

        [spec] = plan.restores
        assert (spec.exact["forgotten_at"], spec.dropped) == (_LONG_AGO, ["ep-x"])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "source", ["alice", "a2"], ids=["same-entity", "new-entity"]
    )
    async def test_a_new_edge_saying_a_sentence_forgotten_later_is_forgotten(
        self, source: str
    ) -> None:
        """Said before the forget, ingested after: the forget covers it, even
        where a hard forget's entities came back under new uuids."""
        said = _edge("l1", fact=_SENTENCE, source=source)
        plan = await _plan(
            [], [_record(_AFTER_SAID)], _result(said, cites=["l1"]), _state()
        )

        assert Reapply(uuid="l1", hard=False, reason="user_signal") in plan.reapply
        assert "l1" in plan.forgotten

    @pytest.mark.asyncio
    async def test_the_same_sentence_said_after_the_forget_is_kept(self) -> None:
        said = _edge("l1", fact=_SENTENCE)
        plan = await _plan([], [_record()], _result(said, cites=["l1"]), _state())

        assert plan.reapply == [] and "l1" not in plan.forgotten


class TestSnapshot:
    @pytest.mark.asyncio
    async def test_the_snapshot_reads_every_forgotten_fact(self) -> None:
        driver = _driver(_state())

        snapshot = await snapshot_forgotten(driver)

        assert snapshot["f1"].fields["fact_redacted"] == _SENTENCE
        query = driver.execute_query.await_args.args[0]
        assert f"WHERE {forgotten_fact_predicate('e')}" in query

    @pytest.mark.asyncio
    async def test_a_failed_snapshot_guards_nothing_and_ingestion_goes_on(self) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = RuntimeError("down")

        assert await snapshot_forgotten(driver) == {}
