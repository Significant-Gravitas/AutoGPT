"""Unit tests for the forget's side of the dream citation markers
(``recall_forget_writes.py``), through ``recall_forget.retract``: it reads
the graph's markers before anything else and completes the landed ones
(``recall_reconcile.py``, pinned in ``recall_reconcile_test.py``), and
after its cascade asks for a retry (``cleanup_error``) while a write that
could still land cites what it reached. The live runs are
``recall_provenance_integration_test.py`` and
``recall_marker_race_integration_test.py``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from . import recall_forget, recall_forget_writes, recall_reconcile
from .recall_forget_fake import (
    CLEANUP,
    NOTHING_DERIVED,
    SOFT,
    forget,
    in_flight_reads,
    scripted,
)


class TestPendingDreamRecords:
    """Every forget completes the graph's pending dream records first
    (``recall_reconcile.py``, pinned in ``recall_reconcile_test.py``), so
    its cascade sees them."""

    @pytest.mark.asyncio
    async def test_it_reads_them_before_anything_else(self) -> None:
        driver = scripted(*SOFT, *NOTHING_DERIVED)

        await forget(driver, ["u1"])

        first = driver.execute_query.await_args_list[0]
        assert first.args[0] == recall_reconcile.MARKERS_QUERY

    @pytest.mark.asyncio
    async def test_a_failed_read_forgets_anyway_and_asks_for_a_retry(self) -> None:
        driver = scripted(*SOFT, *NOTHING_DERIVED, markers=RuntimeError("down"))

        result = await forget(driver, ["u1"])

        assert result.deleted == ["u1"], "the fact itself is forgotten"
        assert [(f.uuid, f.code) for f in result.failures] == [("u1", CLEANUP)]
        assert "RuntimeError: down" in result.failures[0].reason

    @pytest.mark.asyncio
    async def test_more_than_one_forget_takes_asks_for_a_retry(self) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = [
            (r, [], None) for r in (*SOFT, *NOTHING_DERIVED)
        ]

        with patch.object(
            recall_forget_writes,
            "reconcile",
            AsyncMock(return_value=recall_reconcile.Reconciled(left=True)),
        ):
            result = await forget(driver, ["u1"])

        assert [(f.uuid, f.code) for f in result.failures] == [("u1", CLEANUP)]
        assert "without a record" in result.failures[0].reason

    @pytest.mark.asyncio
    async def test_a_write_in_flight_citing_what_it_reached_asks_for_a_retry(
        self,
    ) -> None:
        """Its write could land after the forget, and only its own settle
        would retract it. The forget reads what it reached: its roots, the
        derived facts it retracted or walked through, the episodes it hid."""

        async def cascade(driver, group_id, roots, now, result, *, erase) -> None:
            result.derived.append("d1")
            result.passed.append("p1")
            result.redacted_episodes.append("dream-ep")

        driver = scripted(*SOFT, in_flight=1)

        with patch.object(recall_forget, "cascade", cascade):
            result = await forget(driver, ["u1"])

        assert result.deleted == ["u1"], "the fact itself is forgotten"
        assert [(f.uuid, f.code) for f in result.failures] == [("u1", CLEANUP)]
        assert "may still land" in result.failures[0].reason
        [read] = in_flight_reads(driver)
        assert (read["facts"], read["episodes"], read["state"]) == (
            ["u1", "d1", "p1"],
            ["ep1", "dream-ep"],
            "pending",
        )

    @pytest.mark.asyncio
    async def test_a_failed_in_flight_read_asks_for_a_retry(self) -> None:
        driver = scripted(*SOFT, *NOTHING_DERIVED, in_flight=RuntimeError("down"))

        result = await forget(driver, ["u1"])

        assert [(f.uuid, f.code) for f in result.failures] == [("u1", CLEANUP)]
        assert "RuntimeError: down" in result.failures[0].reason

    @pytest.mark.asyncio
    async def test_nothing_forgotten_reads_no_write_in_flight(self) -> None:
        driver = scripted([], [])  # the lookup, then nothing names it

        await forget(driver, ["missing"])

        assert in_flight_reads(driver) == []
