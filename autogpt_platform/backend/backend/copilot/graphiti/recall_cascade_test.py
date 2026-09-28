"""Unit tests for ``recall_cascade.cascade`` over a small in-memory graph
that answers the cascade's queries as FalkorDB would (``_Graph``).

Pin the transitive walk, what a derived fact is retracted with, the root its
reason names, resuming an earlier try, the bounds and failure reporting. The
Cypher itself runs on FalkorDB in ``recall_cascade_integration_test.py``.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, Field

from . import recall_cascade, recall_hide
from .memory_model import ForgetResult, MemoryForgetFailureCode
from .recall import live_fact_predicate

_NOW = "2026-09-28T00:00:00+00:00"
_CLEANUP = MemoryForgetFailureCode.CLEANUP_ERROR


class _Fact(BaseModel):
    facts: list[str] = Field(default_factory=list)
    episodes: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    live: bool = True
    reason: str | None = None


class _Episode(BaseModel):
    cites: list[str] = Field(default_factory=list)  # entity_edges
    facts: list[str] | None = None  # derived_from_facts: a dream's record
    episodes: list[str] = Field(default_factory=list)
    redacted: bool = False


class _Graph:
    """What the cascade's queries see and change, by query."""

    def __init__(self, facts: dict[str, _Fact], episodes: dict[str, _Episode]):
        self.facts = facts
        self.episodes = episodes
        self.queries: list[str] = []
        self.scrubbed: list[str] = []
        # A query that raises, and facts the dream demotes between the
        # cascade's read and its retraction.
        self.fail_on: str | None = None
        self.demoted_meanwhile: set[str] = set()

    async def execute_query(self, query: str, **params: Any):
        self.queries.append(query)
        if query == self.fail_on:
            raise RuntimeError("down")
        return self._answer(query, params), [], None

    def _answer(self, query: str, p: dict[str, Any]) -> list[dict[str, Any]]:
        if query == recall_cascade.EARLIER_QUERY:
            return [
                {"uuid": uuid, "reason": fact.reason}
                for uuid, fact in self.facts.items()
                if fact.reason in p["reasons"]
            ]
        if query == recall_hide.REDACT_EPISODES_QUERY:
            return self._redact_citing(p["uuids"])
        if query == recall_cascade.DERIVED_FACTS_QUERY:
            return self._derived_facts(p)[: p["limit"]]
        if query == recall_cascade.DERIVED_EPISODES_QUERY:
            return self._derived_episodes(p)[: p["limit"]]
        if query == recall_cascade.RETRACT_QUERY:
            return self._retract(p["targets"])
        if query == recall_cascade.REDACT_DERIVED_QUERY:
            for uuid in p["uuids"]:
                self.episodes[uuid].redacted = True
            return [{"mentioned": []}]
        if query == recall_hide.SCRUB_FACTS_QUERY:
            self.scrubbed.extend(p["uuids"])
            return [{"ends": []}]
        return []

    def _redact_citing(self, uuids: list[str]) -> list[dict[str, Any]]:
        rows = []
        for uuid, episode in sorted(self.episodes.items()):
            via = [x for x in episode.cites if x in uuids]
            if via:
                episode.redacted = True
                rows.append({"uuid": uuid, "via": via})
        return rows

    def _derived_facts(self, p: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for uuid, fact in sorted(self.facts.items()):
            via = [x for x in fact.facts if x in p["facts"]]
            via += [x for x in fact.episodes if x in p["episodes"]]
            stated = [s for s in fact.sources if self.episodes[s].facts is None]
            if via and not stated and uuid not in p["seen"]:
                rows.append({"uuid": uuid, "via": via, "live": fact.live})
        return rows

    def _derived_episodes(self, p: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for uuid, episode in sorted(self.episodes.items()):
            via = [x for x in episode.facts or [] if x in p["facts"]]
            via += [x for x in episode.episodes if x in p["episodes"]]
            if via and uuid not in p["seen"]:
                rows.append({"uuid": uuid, "via": via})
        return rows

    def _retract(self, targets: list[dict[str, str]]) -> list[dict[str, Any]]:
        for uuid in self.demoted_meanwhile:
            self.facts[uuid].live = False
        rows = []
        for target in targets:
            fact = self.facts[target["uuid"]]
            if fact.live:
                fact.live, fact.reason = False, target["reason"]
                rows.append({"uuid": target["uuid"]})
        return rows


def _chain() -> _Graph:
    """User fact ``f`` (said in ``e0``); a consolidation ``c`` citing it (its
    dream episode ``dc``); a proposal ``p`` citing ``c`` (``dp``); a
    proposal ``q`` citing the dream episode ``dp``; and ``u``, a user fact
    that shares ``e0`` but was derived from nothing."""
    return _Graph(
        facts={
            "f": _Fact(live=False, reason="user_signal", sources=["e0"]),
            "u": _Fact(sources=["e0"]),
            "c": _Fact(facts=["f"], episodes=["e0"], sources=["dc"]),
            "p": _Fact(facts=["c"], sources=["dp"]),
            "q": _Fact(episodes=["dp"], sources=["dq"]),
        },
        episodes={
            "e0": _Episode(cites=["f", "u"]),
            "dc": _Episode(cites=["c"], facts=["f"], episodes=["e0"]),
            "dp": _Episode(cites=["p"], facts=["c"]),
            "dq": _Episode(cites=["q"], facts=[], episodes=["dp"]),
        },
    )


async def _cascade(graph: _Graph, roots: list[str]) -> ForgetResult:
    result = ForgetResult(redacted_episodes=["e0"])
    await recall_cascade.cascade(graph, "user_a", roots, _NOW, result)
    return result


def _reasons(graph: _Graph, *uuids: str) -> dict[str, str | None]:
    return {uuid: graph.facts[uuid].reason for uuid in uuids}


class TestTheWalk:
    @pytest.mark.asyncio
    async def test_retracts_what_was_derived_level_by_level_and_nothing_else(
        self,
    ) -> None:
        graph = _chain()

        result = await _cascade(graph, ["f"])

        assert (result.derived, result.failures) == (["c", "p", "q"], [])
        assert _reasons(graph, "c", "p", "q") == dict.fromkeys(
            ["c", "p", "q"], "derived_from_forgotten:f"
        )
        assert graph.facts["u"].live, "a user fact sharing the episode stays"
        assert result.redacted_episodes == ["e0", "dc", "dp", "dq"]
        assert graph.scrubbed == ["c", "p", "q"]

    @pytest.mark.asyncio
    async def test_one_forgotten_source_is_enough(self) -> None:
        graph = _chain()
        graph.facts["g"] = _Fact(sources=["e1"])
        graph.episodes["e1"] = _Episode(cites=["g"])
        graph.facts["c"].facts.append("g")

        result = await _cascade(graph, ["f"])

        assert "c" in result.derived
        assert graph.facts["g"].live

    @pytest.mark.asyncio
    async def test_each_round_writes_its_markers_before_any_clean_up(self) -> None:
        graph = _chain()

        await _cascade(graph, ["f"])

        first_round = graph.queries[4:10]
        assert first_round == [
            recall_cascade.RETRACT_QUERY,
            recall_cascade.REDACT_DERIVED_QUERY,
            recall_hide.SCRUB_FACTS_QUERY,
            recall_hide._ENTITY_KEYS_QUERY,
            recall_hide.REDACT_EPISODES_QUERY,
            recall_cascade.DERIVED_FACTS_QUERY,
        ]

    @pytest.mark.asyncio
    async def test_a_derived_fact_a_user_episode_also_states_is_passed_over(
        self,
    ) -> None:
        """``c`` was also said by the user (``e5`` merged into it): it stays,
        and so does what rests on it; its own dream text is still hidden."""
        graph = _chain()
        graph.facts["c"].sources.append("e5")
        graph.episodes["e5"] = _Episode(cites=["c"])

        result = await _cascade(graph, ["f"])

        assert result.derived == []
        assert all(graph.facts[uuid].live for uuid in ("c", "p", "q"))
        assert graph.episodes["dc"].redacted
        assert not graph.episodes["e5"].redacted

    @pytest.mark.asyncio
    async def test_the_walk_goes_on_through_a_derived_fact_no_longer_live(
        self,
    ) -> None:
        """``c`` was superseded: it is left as it is, but ``p`` and ``q`` rest
        on it and are retracted, and its dream text is hidden."""
        graph = _chain()
        graph.facts["c"].live = False

        result = await _cascade(graph, ["f"])

        assert result.derived == ["p", "q"]
        assert graph.facts["c"].reason is None
        assert graph.episodes["dc"].redacted

    @pytest.mark.asyncio
    async def test_each_retraction_names_the_forgotten_fact_it_descends_from(
        self,
    ) -> None:
        graph = _chain()
        graph.facts["g"] = _Fact(live=False, reason="user_signal", sources=["e1"])
        graph.episodes["e1"] = _Episode(cites=["g"])
        graph.facts["d"] = _Fact(episodes=["e1"], sources=["dd"])
        graph.episodes["dd"] = _Episode(cites=["d"], facts=[], episodes=["e1"])

        await _cascade(graph, ["f", "g"])

        assert _reasons(graph, "c", "d", "p") == {
            "c": "derived_from_forgotten:f",
            "d": "derived_from_forgotten:g",
            "p": "derived_from_forgotten:f",
        }


class TestResuming:
    @pytest.mark.asyncio
    async def test_a_second_forget_goes_on_from_what_the_first_retracted(
        self,
    ) -> None:
        graph = _chain()
        graph.facts["c"].live = False
        graph.facts["c"].reason = "derived_from_forgotten:f"

        result = await _cascade(graph, ["f"])

        assert result.derived == ["p", "q"], "c is not counted twice"
        assert graph.scrubbed[0] == "c", "its clean-up is finished again"
        assert graph.queries[0] == recall_cascade.EARLIER_QUERY


class TestBounds:
    @pytest.mark.asyncio
    async def test_a_chain_deeper_than_the_rounds_is_reported_and_resumed(
        self,
    ) -> None:
        graph = _chain()

        with patch.object(recall_cascade, "CASCADE_MAX_ROUNDS", 2):
            first = await _cascade(graph, ["f"])
            second = await _cascade(graph, ["f"])

        assert first.derived == ["c", "p"]
        assert [(f.uuid, f.code) for f in first.failures] == [("f", _CLEANUP)]
        assert "Forget it again" in first.failures[0].reason
        assert (second.derived, second.failures) == (["q"], [])

    @pytest.mark.asyncio
    async def test_more_derived_items_than_the_budget_are_reported(self) -> None:
        graph = _chain()

        with patch.object(recall_cascade, "CASCADE_MAX_ITEMS", 1):
            result = await _cascade(graph, ["f"])

        assert result.derived == ["c"], "the first item, then it stops"
        assert graph.facts["p"].live and graph.facts["q"].live
        assert [f.code for f in result.failures] == [_CLEANUP]


class TestFailures:
    @pytest.mark.asyncio
    async def test_a_failed_step_is_a_cleanup_error_on_every_root(self) -> None:
        graph = _chain()
        graph.fail_on = recall_hide.SCRUB_FACTS_QUERY
        result = ForgetResult()

        await recall_cascade.cascade(graph, "user_a", ["f", "g"], _NOW, result)

        assert [(f.uuid, f.code) for f in result.failures] == [
            ("f", _CLEANUP),
            ("g", _CLEANUP),
        ]
        assert "RuntimeError: down" in result.failures[0].reason
        assert graph.facts["c"].reason == "derived_from_forgotten:f", "marked first"

    @pytest.mark.asyncio
    async def test_a_fact_that_stopped_being_live_is_passed_through_uncounted(
        self,
    ) -> None:
        """The dream demoted ``c`` between the cascade's read and its write:
        the retraction matches nothing, and the walk goes on through it."""
        graph = _chain()
        graph.demoted_meanwhile = {"c"}

        result = await _cascade(graph, ["f"])

        assert result.derived == ["p", "q"]
        assert graph.facts["c"].reason is None

    @pytest.mark.asyncio
    async def test_no_roots_reads_nothing(self) -> None:
        driver = AsyncMock()

        await recall_cascade.cascade(driver, "user_a", [], _NOW, ForgetResult())

        driver.execute_query.assert_not_awaited()


class TestQueries:
    def test_a_fact_a_user_episode_states_is_never_matched(self) -> None:
        query = recall_cascade.DERIVED_FACTS_QUERY
        assert "stated.derived_from_facts IS NULL" in query
        assert "WHERE independent = 0" in query

    def test_the_retraction_is_a_soft_forgets_on_a_fact_still_live(self) -> None:
        query = recall_cascade.RETRACT_QUERY
        assert live_fact_predicate("e") in query
        assert "e.forgotten_at = coalesce(e.forgotten_at, $now)" in query
        assert "e.expired_at = coalesce(e.expired_at, $now)" in query
        assert "e.expiration_reason = target.reason" in query
        assert "invalid_at" not in query, "a forget is not a world change"

    def test_the_reason_names_the_root(self) -> None:
        assert recall_cascade.derived_reason("f1") == "derived_from_forgotten:f1"
