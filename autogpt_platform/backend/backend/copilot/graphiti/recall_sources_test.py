"""Unit tests for ``recall_sources``: where a source a forget reached leads,
and the walk up the records from a cited derived fact no longer live
(``ancestry``), the cascade's walk-through rule upward, bounded like the
cascade. The live runs are in ``recall_ancestry_integration_test.py``.
"""

from unittest.mock import patch

import pytest

from . import recall_sources
from .recall_cascade import CASCADE_MAX_ROUNDS
from .recall_sources import Reach, ancestry, fact_reach, fact_states
from .recall_sources_fake import episode, fact, reads, sources


async def _walk(driver, cited: list[str]) -> recall_sources.Ancestry:
    """The walk from the cited facts, as the check and the settle start it."""
    return await ancestry(driver, list((await fact_states(driver, cited)).values()))


class TestTheWalk:
    @pytest.mark.asyncio
    async def test_a_derived_fact_no_longer_live_leads_to_the_fact_it_rests_on(
        self,
    ) -> None:
        """``d`` was superseded after the pass read it; ``f`` was forgotten."""
        driver = sources(
            {"d": fact(facts=("f",)), "f": fact(forgotten=True, reason="user_signal")}
        )

        walk = await _walk(driver, ["d"])

        assert walk.reached == {"d": Reach(root="f", hard=False)}
        assert not walk.unfinished

    @pytest.mark.asyncio
    async def test_it_goes_up_level_by_level_and_names_the_root(self) -> None:
        """``f`` was retracted by a cascade from ``root``."""
        driver = sources(
            {
                "d1": fact(facts=("d2",)),
                "d2": fact(facts=("f",)),
                "f": fact(forgotten=True, reason="derived_from_forgotten:root"),
            }
        )

        walk = await _walk(driver, ["d1"])

        assert walk.reached == {"d1": Reach(root="root", hard=False)}
        assert reads(driver, recall_sources.FACT_STATES_QUERY) == [
            ["d1"],
            ["d2"],
            ["f"],
        ]

    @pytest.mark.asyncio
    async def test_a_source_a_hard_forget_reached_leads_there_erasing(self) -> None:
        """``gone`` and ``purged`` were purged; ``h`` is still being purged;
        a hard reach wins over the soft one ``f`` gives."""
        driver = sources(
            {
                "d1": fact(facts=("gone",)),
                "d2": fact(facts=("h",)),
                "d3": fact(facts=("f", "purged")),
                "h": fact(forgotten=True, hard=True),
                "f": fact(forgotten=True),
            }
        )

        walk = await _walk(driver, ["d1", "d2", "d3"])

        assert walk.reached == {
            "d1": Reach(root="gone", hard=True),
            "d2": Reach(root="h", hard=True),
            "d3": Reach(root="purged", hard=True),
        }

    @pytest.mark.asyncio
    async def test_a_source_two_cited_facts_share_is_read_once_for_the_first(
        self,
    ) -> None:
        """Either caller needs one: the check drops the write, and the
        settle's cascade from any cited fact reaches every fact the write
        made, whose record names them all."""
        driver = sources(
            {
                "d1": fact(facts=("f",)),
                "d2": fact(facts=("f",)),
                "f": fact(forgotten=True),
            }
        )

        walk = await _walk(driver, ["d1", "d2"])

        assert walk.reached == {"d1": Reach(root="f", hard=False)}
        assert reads(driver, recall_sources.FACT_STATES_QUERY) == [["d1", "d2"], ["f"]]

    @pytest.mark.asyncio
    async def test_a_hidden_source_episode_leads_to_the_root_it_was_hidden_for(
        self,
    ) -> None:
        driver = sources(
            {
                "d1": fact(episodes=("turn",)),
                "d2": fact(episodes=("tomb",)),
                "f": fact(forgotten=True),
            },
            {
                "turn": episode(hidden=True, hidden_for=("f",)),
                "tomb": episode(hidden=True, hard=True),
            },
        )

        walk = await _walk(driver, ["d1", "d2"])

        assert walk.reached == {
            "d1": Reach(root="f", hard=False),
            "d2": Reach(root="tomb", hard=True),
        }

    @pytest.mark.asyncio
    async def test_a_live_fact_a_users_fact_or_a_recallable_episode_ends_it_clean(
        self,
    ) -> None:
        """A live source would have been retracted with its own source; a
        fact a user states, and what rests on it, a forget passes over."""
        driver = sources(
            {
                "d": fact(facts=("live", "users"), episodes=("turn",)),
                "live": fact(live=True, facts=("f",)),
                "users": fact(derived=False, facts=("f",)),
                "f": fact(forgotten=True),
            },
            {"turn": episode()},
        )

        walk = await _walk(driver, ["d"])

        assert (walk.reached, walk.unfinished) == ({}, False)
        assert reads(driver, recall_sources.FACT_STATES_QUERY) == [
            ["d"],
            ["live", "users"],
        ]

    @pytest.mark.asyncio
    async def test_only_a_derived_fact_no_longer_live_starts_it(self) -> None:
        """A live fact, one already forgotten (the check before the walk
        reads that) and a user's own fact read once, and nothing more."""
        driver = sources(
            {
                "live": fact(live=True, facts=("f",)),
                "gone-already": fact(forgotten=True, facts=("f",)),
                "users": fact(derived=False, facts=("f",)),
            }
        )

        walk = await _walk(driver, ["live", "gone-already", "users"])

        assert walk.reached == {}
        assert driver.execute_query.await_count == 1


class TestTheBound:
    @pytest.mark.asyncio
    async def test_a_chain_deeper_than_the_cascades_rounds_is_unfinished(
        self,
    ) -> None:
        depth = CASCADE_MAX_ROUNDS + 1
        chain = {f"d{i}": fact(facts=(f"d{i + 1}",)) for i in range(depth)}
        driver = sources({**chain, f"d{depth}": fact(forgotten=True)})

        walk = await _walk(driver, ["d0"])

        assert (walk.reached, walk.unfinished) == ({}, True)
        assert len(reads(driver, recall_sources.FACT_STATES_QUERY)) == 1 + (
            CASCADE_MAX_ROUNDS
        )

    @pytest.mark.asyncio
    async def test_more_sources_than_the_cascades_budget_is_unfinished(
        self,
    ) -> None:
        driver = sources({"d": fact(facts=("p1", "p2", "p3"))})

        with patch.object(recall_sources, "CASCADE_MAX_ITEMS", 2):
            walk = await _walk(driver, ["d"])

        assert walk.unfinished
        assert reads(driver, recall_sources.FACT_STATES_QUERY) == [["d"]]


class TestReach:
    def test_a_fact_gone_forgotten_or_still_there(self) -> None:
        assert fact_reach("gone", None) == Reach(root="gone", hard=True)
        forgotten = {"uuid": "f", **fact(forgotten=True, reason="user_signal")}
        assert fact_reach("f", forgotten) == Reach(root="f", hard=False)
        assert fact_reach("d", {"uuid": "d", **fact()}) is None

    def test_the_fact_read_carries_the_cascades_walk_through_rule(self) -> None:
        query = recall_sources.FACT_STATES_QUERY
        assert "stated.derived_from_facts IS NULL" in query
        assert (
            "e.derived_from_facts IS NOT NULL AND independent = 0 AS derived" in query
        )
        assert "e.hard_forgotten_at IS NOT NULL" in query

    def test_the_reads_write_nothing(self) -> None:
        for query in (
            recall_sources.FACT_STATES_QUERY,
            recall_sources.EPISODE_STATES_QUERY,
        ):
            assert "SET" not in query and "DELETE" not in query
