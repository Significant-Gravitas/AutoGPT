"""Unit tests for the cascade a forget runs (``recall_forget.retract``):
on what it retracted, erasing for a hard forget, never after a failed
hide, and resumed from a purged root something still names. The cascade
itself is pinned in ``recall_cascade_test.py``; the live runs are
``recall_cascade_integration_test.py`` and
``recall_cascade_resume_integration_test.py``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from . import recall_forget
from .memory_model import MemoryForgetFailureCode
from .recall_forget_fake import CLEANUP, SOFT, forget, own_call, scripted


class TestCascade:
    """The forget hands the facts it retracted and hid to
    ``recall_cascade.cascade`` (pinned in ``recall_cascade_test.py``)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("hard", [False, True], ids=["soft", "hard"])
    async def test_it_runs_on_what_was_retracted_and_its_count_is_returned(
        self, hard: bool
    ) -> None:
        """A hard forget's cascade erases the text it reaches as well."""
        seen: list[tuple[list[str], bool]] = []

        async def cascade(driver, group_id, roots, now, result, *, erase) -> None:
            seen.append((roots, erase))
            result.derived.extend(["d1", "d2"])

        purged = ([], [], [{"uuid": "u1", "deleted_entities": []}]) if hard else ()
        driver = scripted(*SOFT, *purged)

        with patch.object(recall_forget, "cascade", cascade):
            result = await forget(driver, ["u1"], hard=hard)

        assert seen == [(["u1"], hard)]
        assert (result.deleted, result.derived) == (["u1"], ["d1", "d2"])

    @pytest.mark.asyncio
    async def test_a_failed_hide_leaves_it_for_the_next_forget(self) -> None:
        driver = scripted([{"uuid": "u1"}], [{"uuid": "u1"}], RuntimeError("down"))
        cascade = AsyncMock()

        with patch.object(recall_forget, "cascade", cascade):
            result = await forget(driver, ["u1"])

        cascade.assert_not_awaited()
        assert [(f.uuid, f.code) for f in result.failures] == [("u1", CLEANUP)]

    @pytest.mark.asyncio
    async def test_a_purged_root_something_names_resumes_its_cascade_erasing(
        self,
    ) -> None:
        """``gone`` was purged by a hard forget whose cascade stopped short; a
        record still names it. ``never`` names nothing and stays no match."""
        seen: list[tuple[list[str], bool]] = []

        async def cascade(driver, group_id, roots, now, result, *, erase) -> None:
            seen.append((roots, erase))
            result.derived.append("d1")

        driver = scripted([], [{"uuid": "gone"}])  # the lookup, the names

        with patch.object(recall_forget, "cascade", cascade):
            result = await forget(driver, ["gone", "never"])

        assert seen == [(["gone"], True)], "the root is gone: it was hard"
        assert (result.resumed, result.derived, result.deleted) == (
            ["gone"],
            ["d1"],
            [],
        )
        assert [(f.uuid, f.code) for f in result.failures] == [
            ("never", MemoryForgetFailureCode.NO_MATCH)
        ]
        _, names = own_call(driver, 1)
        assert names == {
            "uuids": ["gone", "never"],
            "prefix": "derived_from_forgotten:",
        }
