"""Tests for graphiti_forget delete helpers."""

from unittest.mock import AsyncMock

import pytest

from backend.copilot.tools.graphiti_forget import _hard_delete_edges, _soft_delete_edges


class TestSoftDeleteOverReportsSuccess:
    """_soft_delete_edges always appends UUID to deleted list even when
    the Cypher MATCH found no edge (query succeeds but matches nothing).
    """

    @pytest.mark.asyncio
    async def test_reports_failure_when_no_edge_matched(self) -> None:
        driver = AsyncMock()
        # execute_query returns empty result set — no edge matched
        driver.execute_query.return_value = ([], None, None)

        deleted, failed = await _soft_delete_edges(
            driver, ["nonexistent-uuid"], "test-user"
        )
        # Should NOT report success when nothing was actually updated
        assert deleted == [], f"over-reported success: {deleted}"
        assert failed == ["nonexistent-uuid"]


class TestSoftDeleteNoMatchReportsFailure:
    """When the query returns empty records (no edge with that UUID exists
    in the database), _soft_delete_edges should report it as failed.
    """

    @pytest.mark.asyncio
    async def test_soft_delete_handles_non_relates_to_edge(self) -> None:
        driver = AsyncMock()
        # Simulate: RELATES_TO match returns nothing (edge is MENTIONS type)
        driver.execute_query.return_value = ([], None, None)

        deleted, failed = await _soft_delete_edges(
            driver, ["mentions-edge-uuid"], "test-user"
        )
        # With the bug, this reports success even though nothing was updated
        assert "mentions-edge-uuid" not in deleted


class TestHardDeleteBasicFlow:
    """Verify _hard_delete_edges calls the right queries."""

    @pytest.mark.asyncio
    async def test_hard_delete_calls_both_queries(self) -> None:
        driver = AsyncMock()
        # First call (delete) returns a matched record, second (cleanup) returns empty
        driver.execute_query.side_effect = [
            ([{"uuid": "uuid-1"}], None, None),
            ([], None, None),
        ]

        deleted, failed = await _hard_delete_edges(driver, ["uuid-1"], "test-user")
        assert deleted == ["uuid-1"]
        assert failed == []
        # Should call: 1) delete edge, 2) clean episode back-refs
        assert driver.execute_query.call_count == 2

    @pytest.mark.asyncio
    async def test_hard_delete_reports_failure_when_no_edge_matched(self) -> None:
        driver = AsyncMock()
        # Delete query returns no records — edge not found
        driver.execute_query.return_value = ([], None, None)

        deleted, failed = await _hard_delete_edges(
            driver, ["nonexistent-uuid"], "test-user"
        )
        assert deleted == []
        assert failed == ["nonexistent-uuid"]
        # Only the delete query should run — cleanup skipped
        assert driver.execute_query.call_count == 1
