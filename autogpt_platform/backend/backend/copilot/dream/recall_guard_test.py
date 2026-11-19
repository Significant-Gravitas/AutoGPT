"""The recall guard's policy: where its window comes from, and which reasons
still demote a recently recalled fact. The statements that apply it are
tested with the writers (``graphiti/guarded_writes_test.py``) and on FalkorDB
(``graphiti/recall_guard_integration_test.py``)."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from backend.copilot.graphiti.recall_stamp import RecallProtection, stamp_time
from backend.util.settings import Config

from . import recall_guard
from .recall_guard import DemotionGuard, demotion_protect_window

_NOW = datetime(2026, 9, 28, 3, 0, tzinfo=timezone.utc)
_CITABLE = {"hot", "other"}


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


def _overrides(protection: RecallProtection, edge_uuid: str) -> bool:
    """Whether *protection*'s override reaches *edge_uuid*, as the statement's
    ``NOT ($override AND ($cited IS NULL OR e.uuid <> $cited))`` reads it."""
    return protection.override and (
        protection.cited is None or edge_uuid != protection.cited
    )


class TestTheWindow:
    def test_the_default_is_thirty_days_and_described_as_a_floor(self) -> None:
        field = Config.model_fields["dream_demotion_protect_days"]

        assert field.default == 30
        description = field.description or ""
        assert "floor" in description
        assert "0 turns this guard off" in description
        assert "does not change the sanitize prompt" in description

    def test_the_window_is_read_from_settings(self, window) -> None:
        window(7)

        assert demotion_protect_window() == timedelta(days=7)

    def test_the_real_settings_carry_the_window(self) -> None:
        assert demotion_protect_window() == timedelta(
            days=Config().dream_demotion_protect_days
        )

    def test_the_guard_starts_the_window_that_many_days_back(self, window) -> None:
        window(30)

        guard = DemotionGuard.at(_NOW, _CITABLE)

        assert guard.recalled_since == stamp_time(_NOW - timedelta(days=30))
        assert guard.citable == frozenset(_CITABLE)

    def test_zero_turns_the_guard_off(self, window) -> None:
        window(0)

        guard = DemotionGuard.at(_NOW, _CITABLE)

        assert guard.recalled_since is None
        assert guard.protection("stale_fact").params()["recalled_since"] is None


class TestWhatStillDemotesARecalledFact:
    def test_the_users_retraction_overrides(self, window) -> None:
        window(30)
        protection = DemotionGuard.at(_NOW, _CITABLE).protection("user_signal")

        assert _overrides(protection, "hot")

    @pytest.mark.parametrize(
        "reason", ["contradicted_by:other", "contradicted_by:  other  "]
    )
    def test_a_contradiction_by_another_fact_the_pass_read_overrides(
        self, window, reason: str
    ) -> None:
        window(30)
        protection = DemotionGuard.at(_NOW, _CITABLE).protection(reason)

        assert protection.cited == "other"
        assert _overrides(protection, "hot")

    def test_a_fact_cannot_contradict_itself(self, window) -> None:
        """The demoted edge's own uuid is listed to the model, so it is the one
        citation an injected reason could always make."""
        window(30)
        protection = DemotionGuard.at(_NOW, _CITABLE).protection("contradicted_by:hot")

        assert not _overrides(protection, "hot")
        assert _overrides(protection, "other"), "it still reaches a neighbour"

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
            "stale_fact contradicted_by:other",
            "user_signal\nstale_fact",
            "USER_SIGNAL",
        ],
    )
    def test_every_other_reason_is_no_override(self, window, reason: str) -> None:
        window(30)
        protection = DemotionGuard.at(_NOW, _CITABLE).protection(reason)

        assert not _overrides(protection, "hot")
        assert protection.recalled_since is not None

    def test_a_pass_that_read_nothing_lets_no_contradiction_override(
        self, window
    ) -> None:
        window(30)
        protection = DemotionGuard.at(_NOW, ()).protection("contradicted_by:other")

        assert not _overrides(protection, "hot")
