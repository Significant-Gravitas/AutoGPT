"""Tests for graphiti_forget delete helpers."""

from unittest.mock import AsyncMock

import pytest

from backend.copilot.tools.graphiti_forget import (
    _hard_delete_edges,
    _retract_edges,
    _soft_delete_edges,
    invalidate_entity_direct_neighbors,
    mark_edges_superseded,
)


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


class TestRetractEdgesSnodgrass:
    """`_retract_edges` is the system-retraction soft delete — must set
    ONLY `expired_at`, never `invalid_at`. Conflating the two breaks the
    bi-temporal model (see graphiti audit §6.13)."""

    @pytest.mark.asyncio
    async def test_retract_sets_only_expired_at(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = ([{"uuid": "u1"}], None, None)

        await _retract_edges(driver, ["u1"], "test-user")

        query = driver.execute_query.call_args.args[0]
        assert "e.expired_at = $now" in query
        # ``now`` parameter is bound from Python (FalkorDB doesn't
        # implement Cypher's no-arg ``datetime()``).
        assert "now" in driver.execute_query.call_args.kwargs
        # Critical contract: must NOT touch invalid_at
        assert "invalid_at" not in query

    @pytest.mark.asyncio
    async def test_retract_reports_failure_on_no_match(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = ([], None, None)

        deleted, failed = await _retract_edges(driver, ["missing"], "test-user")
        assert deleted == []
        assert failed == ["missing"]


class TestSoftDeleteContradictionPath:
    """`_soft_delete_edges` is reserved for the contradiction detector
    and MUST still set both expired_at AND invalid_at."""

    @pytest.mark.asyncio
    async def test_soft_delete_sets_both_timestamps(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = ([{"uuid": "u1"}], None, None)

        await _soft_delete_edges(driver, ["u1"], "test-user")

        query = driver.execute_query.call_args.args[0]
        assert "e.invalid_at = $now" in query
        assert "e.expired_at = $now" in query
        # ``now`` parameter is bound from Python (FalkorDB doesn't
        # implement Cypher's no-arg ``datetime()``).
        assert "now" in driver.execute_query.call_args.kwargs


class TestMarkEdgesSuperseded:
    @pytest.mark.asyncio
    async def test_sets_status_and_reason(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = ([{"uuid": "u1"}], None, None)

        deleted, failed = await mark_edges_superseded(
            driver,
            ["u1"],
            reason="stale_fact",
            new_status="superseded",
            user_id="abc",
        )

        assert deleted == ["u1"]
        assert failed == []
        call_kwargs = driver.execute_query.call_args.kwargs
        assert call_kwargs["new_status"] == "superseded"
        assert call_kwargs["reason"] == "stale_fact"
        query = driver.execute_query.call_args.args[0]
        assert "e.status = $new_status" in query
        assert "e.expiration_reason = $reason" in query
        assert "e.expired_at = $now" in query
        # ``now`` parameter is bound from Python (FalkorDB doesn't
        # implement Cypher's no-arg ``datetime()``).
        assert "now" in driver.execute_query.call_args.kwargs

    @pytest.mark.asyncio
    async def test_default_status_is_superseded(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = ([{"uuid": "u1"}], None, None)
        await mark_edges_superseded(driver, ["u1"], reason="x")
        assert driver.execute_query.call_args.kwargs["new_status"] == "superseded"

    @pytest.mark.asyncio
    async def test_contradicted_status_supported(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = ([{"uuid": "u1"}], None, None)
        await mark_edges_superseded(
            driver, ["u1"], reason="x", new_status="contradicted"
        )
        assert driver.execute_query.call_args.kwargs["new_status"] == "contradicted"

    @pytest.mark.asyncio
    async def test_group_id_scopes_the_match_predicate(self) -> None:
        """Defense-in-depth: when the caller supplies group_id, the Cypher
        MATCH must require it alongside the uuid so a wrong-driver caller
        can't touch another user's edges."""
        driver = AsyncMock()
        driver.execute_query.return_value = ([{"uuid": "u1"}], None, None)

        deleted, failed = await mark_edges_superseded(
            driver,
            ["u1"],
            reason="stale_fact",
            user_id="abc",
            group_id="user_abc",
        )

        assert deleted == ["u1"]
        assert failed == []
        query = driver.execute_query.call_args.args[0]
        assert "{uuid: $uuid, group_id: $group_id}" in query
        assert driver.execute_query.call_args.kwargs["group_id"] == "user_abc"

    @pytest.mark.asyncio
    async def test_no_group_id_keeps_unscoped_match_for_ratification(self) -> None:
        """Omitting group_id preserves the original uuid-only predicate —
        ratification.py still calls without it (per-group driver), so the
        param must stay optional and default to no group filter."""
        driver = AsyncMock()
        driver.execute_query.return_value = ([{"uuid": "u1"}], None, None)

        await mark_edges_superseded(driver, ["u1"], reason="unratified")

        query = driver.execute_query.call_args.args[0]
        assert "{uuid: $uuid}" in query
        assert "group_id" not in query
        assert "group_id" not in driver.execute_query.call_args.kwargs


class TestInvalidateEntityDirectNeighbors:
    """Single-hop demotion. The instinct to write [r:RELATES_TO*1..N] is
    exactly the runaway-demotion bug. This test pins single-hop discipline."""

    @pytest.mark.asyncio
    async def test_single_hop_pattern_in_cypher(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = (
            [{"edge_uuid": "e1"}, {"edge_uuid": "e2"}],
            None,
            None,
        )

        result = await invalidate_entity_direct_neighbors(
            driver, group_id="user_x", entity_uuid="entity-1", reason="dead_client"
        )

        assert result == ["e1", "e2"]
        query = driver.execute_query.call_args.args[0]
        # MUST be single-hop: bare relationship, no quantifier
        assert "[r:RELATES_TO]" in query
        # MUST NOT be multi-hop: variable-length pattern would propagate
        assert "*1.." not in query
        assert "*0.." not in query
        # MUST set status + reason for audit trail
        assert "r.status = 'superseded'" in query
        assert "r.expiration_reason = $reason" in query

    @pytest.mark.asyncio
    async def test_returns_distinct_edge_uuids(self) -> None:
        """The undirected -[r]- pattern can yield the same edge from both
        traversal directions; without DISTINCT the duplicate uuids inflate
        the demotion counts in DreamPassResult / the admin UI."""
        driver = AsyncMock()
        driver.execute_query.return_value = ([{"edge_uuid": "e1"}], None, None)

        await invalidate_entity_direct_neighbors(
            driver, group_id="user_x", entity_uuid="entity-1", reason="dup_check"
        )

        query = driver.execute_query.call_args.args[0]
        assert "RETURN DISTINCT r.uuid AS edge_uuid" in query

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = RuntimeError("boom")

        result = await invalidate_entity_direct_neighbors(
            driver, group_id="user_x", entity_uuid="entity-1", reason="x"
        )
        assert result == []
