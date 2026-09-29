"""Unit tests for ``recall_landing.settle``: a dream write that landed after
a forget reached what it cites is cascaded from before its marker clears.
A forgotten fact softly, under the root its reason names; one a hard forget
reached (purged, still being purged, or erased by its cascade) erasing; a
hidden episode under the root it was hidden for, erasing when a hard forget
emptied it, reached that root, or it is gone; a cited derived fact no
longer live that rests on any of those, from itself under that root, and
from itself under its own name when the walk up stopped at its bound (it
fails closed). The walk is pinned in ``recall_sources_test.py``; the live
runs are in ``recall_marker_integration_test.py``,
``recall_marker_race_integration_test.py`` and
``recall_ancestry_integration_test.py``.
"""

from unittest.mock import patch

import pytest

from . import recall_landing, recall_sources
from .memory_model import MemoryForgetFailure
from .recall_citations import Citations
from .recall_sources_fake import episode, fact, sources


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
    with patch.object(recall_landing, "cascade", cascades):
        return await recall_landing.settle(driver, "user_a", citations)


@pytest.mark.asyncio
async def test_nothing_forgotten_settles_at_once() -> None:
    driver = sources({"f1": fact(live=True)}, {"ep1": episode()})
    cascades = _Cascades()

    settled = await _settle(
        driver, Citations(fact_uuids=["f1"], episode_uuids=["ep1"]), cascades
    )

    assert settled is True
    assert cascades == []


@pytest.mark.asyncio
async def test_a_forgotten_fact_is_cascaded_from_under_its_root() -> None:
    """``f1`` was retracted by a cascade from ``root``; ``f2`` is live."""
    driver = sources(
        {
            "f1": fact(forgotten=True, reason="derived_from_forgotten:root"),
            "f2": fact(live=True),
        }
    )
    cascades = _Cascades()

    settled = await _settle(driver, Citations(fact_uuids=["f1", "f2"]), cascades)

    assert settled is True
    assert cascades == [(["f1"], False, {}, {"f1": "root"})]


@pytest.mark.asyncio
async def test_a_fact_a_hard_forget_reached_is_cascaded_from_erasing() -> None:
    """``gone`` was purged, ``root`` is still being purged, ``d1`` was
    erased by a hard forget's cascade from ``r0``."""
    driver = sources(
        {
            "root": fact(forgotten=True, hard=True, reason="user_signal"),
            "d1": fact(forgotten=True, hard=True, reason="derived_from_forgotten:r0"),
        }
    )
    cascades = _Cascades()

    await _settle(driver, Citations(fact_uuids=["gone", "root", "d1"]), cascades)

    assert cascades == [
        (
            ["gone", "root", "d1"],
            True,
            {},
            {"gone": "gone", "root": "root", "d1": "r0"},
        )
    ]


@pytest.mark.asyncio
async def test_a_hidden_episode_seeds_under_the_root_it_was_hidden_for() -> None:
    """``ep1`` was hidden for ``d1``, which a cascade from ``root``
    retracted; ``tomb`` is emptied, ``orphan`` hidden for a purged fact and
    ``gone`` gone, all three erasing."""
    driver = sources(
        {"d1": fact(forgotten=True, reason="derived_from_forgotten:root")},
        {
            "ep1": episode(hidden=True, hidden_for=("d1",)),
            "ep2": episode(),
            "tomb": episode(hidden=True, hard=True),
            "orphan": episode(hidden=True, hidden_for=("purged",)),
        },
    )
    cascades = _Cascades()

    await _settle(
        driver,
        Citations(episode_uuids=["ep1", "ep2", "tomb", "orphan", "gone"]),
        cascades,
    )

    assert cascades == [
        ([], True, {"gone": "gone", "tomb": "tomb", "orphan": "purged"}, {}),
        ([], False, {"ep1": "root"}, {}),
    ]


@pytest.mark.asyncio
async def test_a_cited_derived_fact_no_longer_live_is_cascaded_from_itself() -> None:
    """``d`` was superseded after the pass read it, and ``f``, which it was
    derived from, forgotten: the cascade from ``d`` leaves it as it is and
    retracts what rests on it, the write's facts among them, under ``f``;
    ``e``'s source was purged, so it erases."""
    driver = sources(
        {
            "d": fact(facts=("f",)),
            "e": fact(facts=("purged",)),
            "f": fact(forgotten=True, reason="user_signal"),
        }
    )
    cascades = _Cascades()

    await _settle(driver, Citations(fact_uuids=["d", "e"]), cascades)

    assert cascades == [
        (["e"], True, {}, {"e": "purged"}),
        (["d"], False, {}, {"d": "f"}),
    ]


@pytest.mark.asyncio
async def test_a_walk_stopped_at_its_bound_fails_closed() -> None:
    """Its sources could not all be read: the cited fact is cascaded from
    under its own name, so the write's facts are retracted all the same."""
    driver = sources({"d": fact(facts=("p1", "p2", "p3"))})
    cascades = _Cascades()

    with patch.object(recall_sources, "CASCADE_MAX_ITEMS", 2):
        await _settle(driver, Citations(fact_uuids=["d"]), cascades)

    assert cascades == [(["d"], False, {}, {"d": "d"})]


@pytest.mark.asyncio
async def test_hard_sources_are_cascaded_from_first() -> None:
    driver = sources({"f1": fact(forgotten=True)})
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
    driver = sources({"f1": fact(forgotten=True)})

    settled = await _settle(driver, Citations(fact_uuids=["f1"]), _Cascades(fail=True))

    assert settled is False
    assert "its marker stays for reconcile" in caplog.text
