"""Unit tests for the cascade a forget runs (``recall_forget.retract``):
on what it retracted, erasing for a hard forget, after a failed hide too,
and resumed from a purged root something still names, or failed as a query
error, the rest of the forget going on, when what names it cannot be read.
The cascade itself is pinned in ``recall_cascade_test.py``; the live runs
are ``recall_cascade_integration_test.py`` and
``recall_cascade_resume_integration_test.py``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from . import recall_forget
from .memory_model import MemoryForgetFailure, MemoryForgetFailureCode
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
    @pytest.mark.parametrize("cascade_fails", [False, True])
    async def test_a_failed_hide_still_retracts_what_the_dream_derived(
        self, cascade_fails: bool
    ) -> None:
        """Sentry's finding: the cascade waited for the hide, so a hide that
        failed left the dream's copies of the fact in recall until the forget
        was repeated. It runs regardless (it redacts the fact's episodes
        itself first), and the fact fails once, however many steps did."""
        driver = scripted([{"uuid": "u1"}], [{"uuid": "u1"}], RuntimeError("down"))

        async def cascade(driver, group_id, roots, now, result, *, erase) -> None:
            result.derived.append("d1")
            if cascade_fails:
                result.failures.append(
                    MemoryForgetFailure.cleanup_error("u1", RuntimeError("down"))
                )

        with patch.object(recall_forget, "cascade", cascade):
            result = await forget(driver, ["u1"])

        assert (result.deleted, result.derived) == (["u1"], ["d1"])
        assert [(f.uuid, f.code) for f in result.failures] == [("u1", CLEANUP)]

    @pytest.mark.asyncio
    async def test_a_hard_forget_whose_hide_failed_cascades_but_purges_nothing(
        self,
    ) -> None:
        """Its cascade erases what the dream derived; the fact's own purge
        waits for a hide that landed, so the next forget finishes it."""
        driver = scripted([{"uuid": "u1"}], [{"uuid": "u1"}], RuntimeError("down"))
        seen: list[bool] = []

        async def cascade(driver, group_id, roots, now, result, *, erase) -> None:
            seen.append(erase)

        purge = AsyncMock()
        with (
            patch.object(recall_forget, "cascade", cascade),
            patch.object(recall_forget, "purge", purge),
        ):
            result = await forget(driver, ["u1"], hard=True)

        assert seen == [True]
        purge.assert_not_awaited()
        assert [(f.uuid, f.code) for f in result.failures] == [("u1", CLEANUP)]

    @pytest.mark.asyncio
    async def test_a_failed_look_for_purged_facts_fails_only_those(self) -> None:
        """Sentry's finding: the read for purged roots raised out of the
        forget, a raw 500 on the settings page and the uuids it found left
        unforgotten. Now a uuid that matched no edge fails as a query error,
        which a retry may clear, and the rest is forgotten."""
        driver = scripted(SOFT[0], RuntimeError("down"), *SOFT[1:])

        with patch.object(recall_forget, "cascade", AsyncMock()):
            result = await forget(driver, ["u1", "gone"])

        assert (result.deleted, result.resumed) == (["u1"], [])
        assert [(f.uuid, f.code) for f in result.failures] == [
            ("gone", MemoryForgetFailureCode.QUERY_ERROR)
        ]

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
