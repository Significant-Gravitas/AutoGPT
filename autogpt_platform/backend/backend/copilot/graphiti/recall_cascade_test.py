"""Unit tests for ``recall_cascade.cascade`` over a small in-memory graph
that answers the cascade's queries as FalkorDB would
(``recall_cascade_fake.py``).

Pin the transitive walk, what a derived fact is retracted with, the root its
reason names, resuming an earlier try, the bounds and failure reporting
(what a hard forget erases: ``recall_erase_test.py``). The Cypher itself runs
on FalkorDB in ``recall_cascade_integration_test.py``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from . import recall_cascade, recall_hide
from .memory_model import ForgetResult, MemoryForgetFailureCode
from .recall import live_fact_predicate
from .recall_cascade_fake import CascadeGraph, FakeEpisode, FakeFact, chain

_NOW = "2026-09-28T00:00:00+00:00"
_CLEANUP = MemoryForgetFailureCode.CLEANUP_ERROR


async def _cascade(graph: CascadeGraph, roots: list[str]) -> ForgetResult:
    result = ForgetResult(redacted_episodes=["e0"])
    await recall_cascade.cascade(graph, "user_a", roots, _NOW, result)
    return result


def _reasons(graph: CascadeGraph, *uuids: str) -> dict[str, str | None]:
    return {uuid: graph.facts[uuid].reason for uuid in uuids}


class TestTheWalk:
    @pytest.mark.asyncio
    async def test_retracts_what_was_derived_level_by_level_and_nothing_else(
        self,
    ) -> None:
        graph = chain()

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
        graph = chain()
        graph.facts["g"] = FakeFact(sources=["e1"])
        graph.episodes["e1"] = FakeEpisode(cites=["g"])
        graph.facts["c"].facts.append("g")

        result = await _cascade(graph, ["f"])

        assert "c" in result.derived
        assert graph.facts["g"].live

    @pytest.mark.asyncio
    async def test_each_round_writes_its_markers_before_any_clean_up(self) -> None:
        graph = chain()

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
        graph = chain()
        graph.facts["c"].sources.append("e5")
        graph.episodes["e5"] = FakeEpisode(cites=["c"])

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
        graph = chain()
        graph.facts["c"].live = False

        result = await _cascade(graph, ["f"])

        assert result.derived == ["p", "q"]
        assert graph.facts["c"].reason is None
        assert graph.episodes["dc"].redacted

    @pytest.mark.asyncio
    async def test_each_retraction_names_the_forgotten_fact_it_descends_from(
        self,
    ) -> None:
        graph = chain()
        graph.facts["g"] = FakeFact(live=False, reason="user_signal", sources=["e1"])
        graph.episodes["e1"] = FakeEpisode(cites=["g"])
        graph.facts["d"] = FakeFact(episodes=["e1"], sources=["dd"])
        graph.episodes["dd"] = FakeEpisode(cites=["d"], facts=[], episodes=["e1"])

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
        graph = chain()
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
        graph = chain()

        with patch.object(recall_cascade, "CASCADE_MAX_ROUNDS", 2):
            first = await _cascade(graph, ["f"])
            second = await _cascade(graph, ["f"])

        assert first.derived == ["c", "p"]
        assert [(f.uuid, f.code) for f in first.failures] == [("f", _CLEANUP)]
        assert "Forget it again" in first.failures[0].reason
        assert (second.derived, second.failures) == (["q"], [])

    @pytest.mark.asyncio
    async def test_more_derived_items_than_the_budget_are_reported(self) -> None:
        graph = chain()

        with patch.object(recall_cascade, "CASCADE_MAX_ITEMS", 1):
            result = await _cascade(graph, ["f"])

        assert result.derived == ["c"], "the first item, then it stops"
        assert graph.facts["p"].live and graph.facts["q"].live
        assert [f.code for f in result.failures] == [_CLEANUP]

    @pytest.mark.asyncio
    async def test_a_budget_used_up_exactly_still_reports_what_is_left(
        self,
    ) -> None:
        """Each round finds a fact and its dream episode, exactly the budget:
        the next round, cut to nothing, is still reported, and each forget
        again goes one level further."""
        graph = chain()

        with patch.object(recall_cascade, "CASCADE_MAX_ITEMS", 2):
            tries = [await _cascade(graph, ["f"]) for _ in range(3)]

        assert [t.derived for t in tries] == [["c"], ["p"], ["q"]]
        assert [[f.code for f in t.failures] for t in tries] == [
            [_CLEANUP],
            [_CLEANUP],
            [],
        ]


class TestFailures:
    @pytest.mark.asyncio
    async def test_a_failed_step_is_a_cleanup_error_on_every_root(self) -> None:
        graph = chain()
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
        graph = chain()
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
