"""Unit tests for ``recall_landing.settle``: a dream write that landed after
a forget reached what it cites is cascaded from before its marker clears.
A forgotten fact softly, under the root its reason names; one a hard forget
reached (purged, still being purged, or erased by its cascade) erasing; a
hidden episode under the root it was hidden for, erasing when a hard forget
emptied it, reached that root, or it is gone. The live runs are in
``recall_marker_integration_test.py`` and
``recall_marker_race_integration_test.py``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import recall_landing
from .memory_model import MemoryForgetFailure
from .recall_citations import Citations


def _fact(
    uuid: str, *, forgotten: bool = False, hard: bool = False, reason: str | None = None
) -> dict:
    """A ``CITED_FACTS_QUERY`` row."""
    return {"uuid": uuid, "forgotten": forgotten, "hard": hard, "reason": reason}


def _driver(*results) -> MagicMock:
    driver = MagicMock()
    driver.execute_query = AsyncMock(side_effect=[(r, [], None) for r in results])
    return driver


class _Cascades(list):
    """The cascades ``settle`` ran: (roots, erase, seeds, named)."""

    def __init__(self, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail

    async def __call__(
        self, driver, group_id, roots, now, result, *, erase, seeds, named
    ) -> None:
        self.append((roots, erase, seeds, named))
        result.derived.append("late-fact")
        if self.fail:
            result.failures.append(MemoryForgetFailure.derived_left(roots[0]))


async def _settle(driver, citations: Citations, cascades: _Cascades) -> bool:
    with (
        patch.object(
            recall_landing,
            "rests_on_a_forget",
            AsyncMock(return_value="cites a fact that is forgotten or gone"),
        ),
        patch.object(recall_landing, "cascade", cascades),
    ):
        return await recall_landing.settle(driver, "user_a", citations)


@pytest.mark.asyncio
async def test_nothing_forgotten_settles_at_once() -> None:
    driver = _driver()
    cascades = _Cascades()

    with (
        patch.object(recall_landing, "rests_on_a_forget", AsyncMock(return_value=None)),
        patch.object(recall_landing, "cascade", cascades),
    ):
        settled = await recall_landing.settle(
            driver, "user_a", Citations(fact_uuids=["f1"])
        )

    assert settled is True
    assert cascades == []
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_forgotten_fact_is_cascaded_from_under_its_root() -> None:
    """``f1`` was retracted by a cascade from ``root``; ``f2`` is live."""
    driver = _driver(
        [
            _fact("f1", forgotten=True, reason="derived_from_forgotten:root"),
            _fact("f2"),
        ]
    )
    cascades = _Cascades()

    settled = await _settle(driver, Citations(fact_uuids=["f1", "f2"]), cascades)

    assert settled is True
    assert cascades == [(["f1"], False, {}, {"f1": "root"})]
    [read] = driver.execute_query.await_args_list
    assert read.args[0] == recall_landing.CITED_FACTS_QUERY
    assert read.kwargs == {"uuids": ["f1", "f2"]}


@pytest.mark.asyncio
async def test_a_purged_fact_is_cascaded_from_erasing() -> None:
    """The pass read it, so a fact gone was purged by a hard forget."""
    driver = _driver([_fact("f1")])
    cascades = _Cascades()

    await _settle(driver, Citations(fact_uuids=["f1", "gone"]), cascades)

    assert cascades == [(["gone"], True, {}, {"gone": "gone"})]


@pytest.mark.asyncio
async def test_a_hidden_episode_seeds_under_the_root_it_was_hidden_for() -> None:
    driver = _driver(
        [
            {"uuid": "ep1", "hidden": True, "hard": False, "hidden_for": ["d1"]},
            {"uuid": "ep2", "hidden": False, "hard": False, "hidden_for": []},
        ],
        [_fact("d1", forgotten=True, reason="derived_from_forgotten:root")],
    )
    cascades = _Cascades()

    await _settle(driver, Citations(episode_uuids=["ep1", "ep2"]), cascades)

    assert cascades == [([], False, {"ep1": "root"}, {})]
    episodes, facts = driver.execute_query.await_args_list
    assert episodes.args[0] == recall_landing.CITED_EPISODES_QUERY
    assert (facts.args[0], facts.kwargs) == (
        recall_landing.CITED_FACTS_QUERY,
        {"uuids": ["d1"]},
    )


@pytest.mark.asyncio
async def test_an_emptied_or_gone_episode_seeds_erasing() -> None:
    """A tombstone, one gone, and one hidden for a fact since purged."""
    driver = _driver(
        [
            {"uuid": "tomb", "hidden": True, "hard": True, "hidden_for": []},
            {"uuid": "orphan", "hidden": True, "hard": False, "hidden_for": ["purged"]},
        ],
        [],
    )
    cascades = _Cascades()

    await _settle(driver, Citations(episode_uuids=["tomb", "orphan", "gone"]), cascades)

    assert cascades == [
        ([], True, {"gone": "gone", "tomb": "tomb", "orphan": "purged"}, {})
    ]


@pytest.mark.asyncio
async def test_a_fact_a_hard_forget_reached_is_cascaded_from_erasing() -> None:
    """``root`` is still being purged (``hard_forgotten_at``); ``d1`` is a
    derived fact its cascade erased; the episode was hidden for ``root``."""
    driver = _driver(
        [{"uuid": "ep1", "hidden": True, "hard": False, "hidden_for": ["root"]}],
        [
            _fact("root", forgotten=True, hard=True, reason="user_signal"),
            _fact("d1", forgotten=True, hard=True, reason="derived_from_forgotten:r0"),
        ],
    )
    cascades = _Cascades()

    await _settle(
        driver, Citations(fact_uuids=["root", "d1"], episode_uuids=["ep1"]), cascades
    )

    assert cascades == [
        (["root", "d1"], True, {"ep1": "root"}, {"root": "root", "d1": "r0"})
    ]


@pytest.mark.asyncio
async def test_hard_sources_are_cascaded_from_first() -> None:
    driver = _driver([_fact("f1", forgotten=True)])
    cascades = _Cascades()

    await _settle(driver, Citations(fact_uuids=["gone", "f1"]), cascades)

    assert [(roots, erase) for roots, erase, _, _ in cascades] == [
        (["gone"], True),
        (["f1"], False),
    ]


@pytest.mark.asyncio
async def test_a_cascade_that_stops_short_keeps_the_marker(
    caplog: pytest.LogCaptureFixture,
) -> None:
    driver = _driver([_fact("f1", forgotten=True)])

    settled = await _settle(driver, Citations(fact_uuids=["f1"]), _Cascades(fail=True))

    assert settled is False
    assert "its marker stays for reconcile" in caplog.text


def test_the_reads_write_nothing() -> None:
    for query in (
        recall_landing.CITED_FACTS_QUERY,
        recall_landing.CITED_EPISODES_QUERY,
    ):
        assert "SET" not in query and "DELETE" not in query


def test_a_hard_forget_is_read_off_its_stamp_or_an_erased_copy() -> None:
    query = recall_landing.CITED_FACTS_QUERY
    assert "e.hard_forgotten_at IS NOT NULL" in query
    assert "coalesce(e.fact_redacted, '-') = ''" in query
