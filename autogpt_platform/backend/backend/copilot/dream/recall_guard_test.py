"""The recall guard's rules: which facts a recall protects, which reasons
still demote a protected fact, where its window comes from, and how its reads
of the graph fail."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.copilot.graphiti.recall_stamp import RecallStamp, stamp_time
from backend.util.settings import Config

from . import recall_guard
from .recall_guard import (
    demotion_protect_window,
    drop_protected,
    guard_at_apply,
    overrides_protection,
    protected_neighbours,
    protected_uuids,
)
from .schemas import DreamDemotion, EntityInvalidation

_NOW = datetime(2026, 9, 28, 3, 0, tzinfo=timezone.utc)
_WINDOW = timedelta(days=30)


def _stamp(uuid: str, *, days_ago: float | None, raw: str | None = None) -> RecallStamp:
    last = raw
    if days_ago is not None:
        last = stamp_time(_NOW - timedelta(days=days_ago))
    return RecallStamp(
        uuid=uuid, recall_count=1 if last else None, last_recalled_at=last
    )


def _demote(uuid: str, reason: str = "stale_fact") -> DreamDemotion:
    return DreamDemotion(edge_uuid=uuid, reason=reason)


@pytest.fixture
def window(mocker):
    """Set ``Config.dream_demotion_protect_days`` as the guard reads it."""

    def _set(days: int) -> None:
        mocker.patch.object(
            recall_guard,
            "Settings",
            return_value=SimpleNamespace(
                config=SimpleNamespace(dream_demotion_protect_days=days)
            ),
        )

    return _set


class TestWhatARecallProtects:
    def test_a_recall_within_the_window_protects_its_fact(self) -> None:
        stamps = [
            _stamp("yesterday", days_ago=1),
            _stamp("edge", days_ago=30),
            _stamp("outside", days_ago=30.01),
            _stamp("never", days_ago=None),
        ]

        assert protected_uuids(stamps, now=_NOW, window=_WINDOW) == {
            "yesterday",
            "edge",
        }

    def test_one_recall_is_enough(self) -> None:
        """Protect-only: even a single incidental recall can only make the
        pass less destructive, so there is no two-recall bar."""
        once = RecallStamp(
            uuid="once", recall_count=1, last_recalled_at=stamp_time(_NOW)
        )

        assert protected_uuids([once], now=_NOW, window=_WINDOW) == {"once"}

    @pytest.mark.parametrize(
        "raw, protected",
        [
            ("2026-09-27T03:00:00Z", True),
            ("2026-09-27T03:00:00", True),
            ("2026-09-29T03:00:00+00:00", True),
            ("not a time", False),
            ("", False),
        ],
    )
    def test_how_a_stamp_is_read(self, raw: str, protected: bool) -> None:
        stamp = _stamp("f", days_ago=None, raw=raw or None)

        assert (protected_uuids([stamp], now=_NOW, window=_WINDOW) == {"f"}) is (
            protected
        )

    def test_a_zero_window_protects_nothing(self) -> None:
        assert (
            protected_uuids(
                [_stamp("yesterday", days_ago=1)], now=_NOW, window=timedelta(0)
            )
            == set()
        )


class TestWhatStillDemotesAProtectedFact:
    def test_the_users_retraction_overrides(self) -> None:
        assert overrides_protection("user_signal", "hot", {"hot", "other"})

    def test_a_contradiction_by_a_fact_the_pass_read_overrides(self) -> None:
        assert overrides_protection("contradicted_by:other", "hot", {"hot", "other"})
        assert overrides_protection(
            "contradicted_by:  other  ", "hot", {"hot", "other"}
        )

    @pytest.mark.parametrize(
        "reason",
        [
            "stale_fact",
            "entity_invalidated:abc",
            "unratified",
            "web_contradicted:https://example.test",
            "no longer seems relevant",
            "contradicted_by:unknown",
            "contradicted_by:",
            "contradicted_by:hot",
            "USER_SIGNAL",
        ],
    )
    def test_every_other_reason_stays_blocked(self, reason: str) -> None:
        assert not overrides_protection(reason, "hot", {"hot", "other"})

    def test_drop_protected_keeps_overrides_and_unprotected(self) -> None:
        demotions = [
            _demote("hot"),
            _demote("hot", "user_signal"),
            _demote("hot", "contradicted_by:other"),
            _demote("hot", "contradicted_by:hot"),
            _demote("cold"),
        ]

        kept, dropped = drop_protected(
            demotions, {"hot"}, {"hot", "other", "cold"}, where="test"
        )

        assert kept == [demotions[1], demotions[2], demotions[4]]
        assert dropped == 2


class TestTheWindowSetting:
    def test_the_default_window_is_thirty_days(self) -> None:
        field = Config.model_fields["dream_demotion_protect_days"]

        assert field.default == 30
        assert "floor" in (field.description or "")

    def test_the_window_is_read_from_settings(self, window) -> None:
        window(7)

        assert demotion_protect_window() == timedelta(days=7)

    def test_the_real_settings_carry_the_window(self) -> None:
        assert demotion_protect_window() == timedelta(
            days=Config().dream_demotion_protect_days
        )

    @pytest.mark.asyncio
    async def test_a_zero_window_turns_the_apply_time_guard_off(self, window) -> None:
        window(0)
        driver = AsyncMock()
        demotions = [_demote("hot")]

        assert await guard_at_apply(driver, "g", "p", demotions, {"hot"}) == (
            demotions,
            0,
        )
        driver.execute_query.assert_not_awaited()


class TestTheApplyTimeRead:
    @pytest.mark.asyncio
    async def test_it_reads_the_targets_now_and_drops_the_recalled(
        self, mocker, window
    ) -> None:
        window(30)
        recalled = RecallStamp(
            uuid="hot",
            recall_count=1,
            last_recalled_at=stamp_time(datetime.now(timezone.utc)),
        )
        read = mocker.patch.object(
            recall_guard, "read_recall_stamps", AsyncMock(return_value=[recalled])
        )
        demotions = [
            _demote("hot"),
            _demote("cold"),
            _demote("wrong", "user_signal"),
        ]

        driver = AsyncMock()

        kept, dropped = await guard_at_apply(
            driver, "user_g", "p-1", demotions, {"hot", "cold", "wrong"}
        )

        assert kept == [demotions[1], demotions[2]]
        assert dropped == 1
        # An override needs no read: its demotion goes ahead either way.
        read.assert_awaited_once_with(driver, "user_g", ["hot", "cold"])

    @pytest.mark.asyncio
    async def test_a_failed_read_keeps_what_the_clamp_let_through(
        self, mocker, window, caplog: pytest.LogCaptureFixture
    ) -> None:
        window(30)
        mocker.patch.object(
            recall_guard, "read_recall_stamps", AsyncMock(return_value=None)
        )
        demotions = [_demote("hot")]

        assert await guard_at_apply(AsyncMock(), "g", "p-2", demotions, {"hot"}) == (
            demotions,
            0,
        )
        assert "could not be read again" in caplog.text


class TestProtectedNeighbours:
    @pytest.mark.asyncio
    async def test_recalled_neighbours_are_protected_unless_the_reason_overrides(
        self, mocker, window
    ) -> None:
        window(30)
        now = stamp_time(datetime.now(timezone.utc))
        stamps = [
            RecallStamp(uuid="recalled", recall_count=2, last_recalled_at=now),
            RecallStamp(uuid="cited", recall_count=1, last_recalled_at=now),
            RecallStamp(uuid="quiet"),
        ]
        mocker.patch.object(
            recall_guard, "read_neighbour_stamps", AsyncMock(return_value=stamps)
        )
        citable = {"recalled", "cited", "quiet"}

        stale = EntityInvalidation(entity_uuid="hub", reason="dead_client")
        retracted = EntityInvalidation(entity_uuid="hub", reason="user_signal")
        contradicted = EntityInvalidation(
            entity_uuid="hub", reason="contradicted_by:cited"
        )

        assert await protected_neighbours(AsyncMock(), "g", "p", stale, citable) == {
            "recalled",
            "cited",
        }
        assert (
            await protected_neighbours(AsyncMock(), "g", "p", retracted, citable)
            == set()
        )
        # A fact cannot contradict itself: the cited neighbour stays protected.
        assert await protected_neighbours(
            AsyncMock(), "g", "p", contradicted, citable
        ) == {"cited"}

    @pytest.mark.asyncio
    async def test_a_failed_read_protects_nothing(
        self, mocker, window, caplog: pytest.LogCaptureFixture
    ) -> None:
        window(30)
        mocker.patch.object(
            recall_guard, "read_neighbour_stamps", AsyncMock(return_value=None)
        )
        invalidation = EntityInvalidation(entity_uuid="hub", reason="dead_client")

        assert (
            await protected_neighbours(AsyncMock(), "g", "p", invalidation, set())
            == set()
        )
        assert "goes ahead unguarded" in caplog.text

    @pytest.mark.asyncio
    async def test_a_zero_window_reads_nothing(self, mocker, window) -> None:
        window(0)
        read = mocker.patch.object(recall_guard, "read_neighbour_stamps", AsyncMock())
        invalidation = EntityInvalidation(entity_uuid="hub", reason="dead_client")

        assert (
            await protected_neighbours(AsyncMock(), "g", "p", invalidation, set())
            == set()
        )
        read.assert_not_awaited()
