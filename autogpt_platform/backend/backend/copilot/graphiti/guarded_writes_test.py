"""The dream's guarded writers against a mocked driver: the Cypher each
sends (single-hop, live facts only, the recall guard in the statement that
writes, one aggregate row for a neighbourhood) and what each reports,
including a write whose outcome is unknown; and the liveness read. ``recall_guard_integration_test.py`` and
``guarded_writes_integration_test.py`` run the same statements on
FalkorDB."""

from unittest.mock import AsyncMock

import pytest

from .guarded_writes import (
    NeighbourWrites,
    WriteOutcome,
    invalidate_entity_direct_neighbors,
    live_fact_uuids,
    supersede_unless_recalled,
)
from .recall import live_fact_predicate
from .recall_stamp import RecallProtection, spared_by_recall


def _outcomes(*rows: tuple[str, bool]) -> tuple[list[dict], None, None]:
    """The one aggregate row the neighbour statement returns."""
    return (
        [{"outcomes": [{"uuid": uuid, "spared": spared} for uuid, spared in rows]}],
        None,
        None,
    )


class TestInvalidateEntityDirectNeighbors:
    """Single-hop demotion. The instinct to write [r:RELATES_TO*1..N] is exactly
    the runaway-demotion bug. This pins single-hop discipline, and the recall
    guard tested per neighbour in the same statement."""

    @pytest.mark.asyncio
    async def test_single_hop_pattern_in_cypher(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = _outcomes(("e1", False), ("e2", False))

        result = await invalidate_entity_direct_neighbors(
            driver, group_id="user_x", entity_uuid="entity-1", reason="dead_client"
        )

        assert result == NeighbourWrites(changed=["e1", "e2"])
        query = driver.execute_query.call_args.args[0]
        # MUST be single-hop: bare relationship, no quantifier
        assert "[r:RELATES_TO]" in query
        # MUST NOT be multi-hop: variable-length pattern would propagate
        assert "*1.." not in query
        assert "*0.." not in query
        # MUST set status + reason for audit trail
        assert "r.status = 'superseded'" in query
        assert "r.expiration_reason = $reason" in query
        # MUST leave a retired or forgotten neighbour's audit fields alone
        assert f"WHERE {live_fact_predicate('r')}" in query
        assert "forgotten_at =" not in query

    @pytest.mark.asyncio
    async def test_the_recall_guard_is_in_the_writing_statement(self) -> None:
        """The guard and the write are one statement: no neighbour recalled
        before it runs can be demoted without an override."""
        driver = AsyncMock()
        driver.execute_query.return_value = _outcomes(("old", False), ("recent", True))
        protection = RecallProtection(
            recalled_since="2026-08-29T03:00:00.000000+00:00",
            override=True,
            cited="c",
        )

        result = await invalidate_entity_direct_neighbors(
            driver,
            group_id="user_x",
            entity_uuid="entity-1",
            reason="contradicted_by:c",
            protection=protection,
        )

        assert result == NeighbourWrites(changed=["old"], spared=["recent"])
        query = driver.execute_query.call_args.args[0]
        params = driver.execute_query.call_args.kwargs
        assert f"WITH r, {spared_by_recall('r')} AS spared" in query
        assert "FOREACH (_ IN CASE WHEN spared THEN [] ELSE [1] END |" in query
        assert query.rstrip().endswith(
            "RETURN collect({uuid: r.uuid, spared: spared}) AS outcomes"
        )
        assert {k: params[k] for k in protection.params()} == protection.params()

    @pytest.mark.asyncio
    async def test_no_protection_spares_nothing(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = ([], None, None)

        await invalidate_entity_direct_neighbors(
            driver, group_id="user_x", entity_uuid="entity-1", reason="x"
        )

        params = driver.execute_query.call_args.kwargs
        assert (params["recalled_since"], params["override"], params["cited"]) == (
            None,
            False,
            None,
        )

    @pytest.mark.asyncio
    async def test_each_edge_is_written_once(self) -> None:
        """The undirected -[r]- pattern can yield the same edge from both
        traversal directions; without DISTINCT the duplicate uuids inflate
        the demotion counts in DreamPassResult / the admin UI."""
        driver = AsyncMock()
        driver.execute_query.return_value = _outcomes(("e1", False))

        await invalidate_entity_direct_neighbors(
            driver, group_id="user_x", entity_uuid="entity-1", reason="dup_check"
        )

        query = driver.execute_query.call_args.args[0]
        assert "WITH DISTINCT r" in query

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reply",
        [([{"outcomes": []}], None, None), ([], None, None), None],
        ids=["empty-aggregate", "no-row", "no-result"],
    )
    async def test_a_neighbourhood_with_no_live_fact_is_acknowledged_empty(
        self, reply
    ) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = reply

        result = await invalidate_entity_direct_neighbors(
            driver, group_id="user_x", entity_uuid="entity-1", reason="x"
        )

        assert result == NeighbourWrites()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure",
        [RuntimeError("boom"), TimeoutError("acknowledgement lost")],
        ids=["error", "lost-acknowledgement"],
    )
    async def test_a_statement_that_raises_has_an_unknown_outcome(
        self, failure: Exception
    ) -> None:
        """It may have committed before the reply was lost: never reported as
        having changed nothing."""
        driver = AsyncMock()
        driver.execute_query.side_effect = failure

        result = await invalidate_entity_direct_neighbors(
            driver, group_id="user_x", entity_uuid="entity-1", reason="x"
        )

        assert result == NeighbourWrites(unknown=True)


class TestSupersedeUnlessRecalled:
    """The dream's demotions: one statement per edge that tests the recall
    guard and writes, and one outcome per requested uuid, in order."""

    @pytest.mark.asyncio
    async def test_one_outcome_per_uuid_in_order(self) -> None:
        answers = {
            "changed": [{"uuid": "changed", "spared": False}],
            "spared": [{"uuid": "spared", "spared": True}],
            "gone": [],
        }

        async def execute(query: str, **params: object):
            return (answers[str(params["uuid"])], None, None)

        driver = AsyncMock()
        driver.execute_query.side_effect = execute

        outcomes = await supersede_unless_recalled(
            driver,
            ["changed", "spared", "gone", "changed"],
            reason="stale_fact",
            new_status="superseded",
            group_id="user_x",
            protection=RecallProtection(
                recalled_since="2026-08-29T03:00:00.000000+00:00"
            ),
        )

        assert outcomes == [
            WriteOutcome.CHANGED,
            WriteOutcome.SPARED,
            WriteOutcome.UNMATCHED,
            WriteOutcome.CHANGED,
        ]

    @pytest.mark.asyncio
    async def test_the_guard_and_the_write_are_one_statement(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = (
            [{"uuid": "a", "spared": False}],
            None,
            None,
        )
        protection = RecallProtection(
            recalled_since="2026-08-29T03:00:00.000000+00:00", override=True
        )

        await supersede_unless_recalled(
            driver,
            ["a"],
            reason="user_signal",
            new_status="contradicted",
            group_id="user_x",
            protection=protection,
        )

        query = driver.execute_query.call_args.args[0]
        params = driver.execute_query.call_args.kwargs
        assert "MATCH ()-[e:RELATES_TO {uuid: $uuid, group_id: $group_id}]->()" in query
        assert f"WHERE {live_fact_predicate('e')}" in query
        assert f"WITH e, {spared_by_recall('e')} AS spared" in query
        assert "FOREACH (_ IN CASE WHEN spared THEN [] ELSE [1] END |" in query
        assert "e.expiration_reason = $reason" in query
        assert query.rstrip().endswith("RETURN e.uuid AS uuid, spared")
        assert params["new_status"] == "contradicted"
        assert params["group_id"] == "user_x"
        assert {k: params[k] for k in protection.params()} == protection.params()

    @pytest.mark.asyncio
    async def test_a_write_that_raises_has_an_unknown_outcome(self) -> None:
        """It may have committed before its reply was lost, so it is neither
        reported as changed nor as having matched nothing."""
        driver = AsyncMock()
        driver.execute_query.side_effect = TimeoutError("acknowledgement lost")

        outcomes = await supersede_unless_recalled(
            driver,
            ["a", "b"],
            reason="stale_fact",
            new_status="superseded",
            group_id="user_x",
            protection=RecallProtection(),
        )

        assert outcomes == [WriteOutcome.UNKNOWN, WriteOutcome.UNKNOWN]


class TestExpectedStatus:
    """Ratification supersedes only a still-tentative proposal: the status
    check is part of the same guarded statement."""

    @pytest.mark.asyncio
    async def test_the_expected_status_is_checked_in_the_statement(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = ([], None, None)

        outcomes = await supersede_unless_recalled(
            driver,
            ["tentative-1"],
            reason="unratified",
            new_status="superseded",
            group_id="user_x",
            protection=RecallProtection(),
            expected_status="tentative",
        )

        assert outcomes == [WriteOutcome.UNMATCHED]
        query = driver.execute_query.call_args.args[0]
        params = driver.execute_query.call_args.kwargs
        assert "AND ($expected_status IS NULL OR e.status = $expected_status)" in query
        assert params["expected_status"] == "tentative"

    @pytest.mark.asyncio
    async def test_no_expected_status_is_sent_as_null(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = (
            [{"uuid": "a", "spared": False}],
            None,
            None,
        )

        await supersede_unless_recalled(
            driver,
            ["a"],
            reason="stale_fact",
            new_status="superseded",
            group_id="user_x",
            protection=RecallProtection(),
        )

        assert driver.execute_query.call_args.kwargs["expected_status"] is None


class TestLiveFactUuids:
    """The read that settles, after a pass's writes, which spared facts are
    still live: one statement over just those facts, one row back."""

    @pytest.mark.asyncio
    async def test_one_bounded_statement_and_one_row(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = ([{"live": ["a", "c"]}], None, None)

        live = await live_fact_uuids(driver, "user_x", ["a", "b", "c"])

        assert live == {"a", "c"}
        query = driver.execute_query.call_args.args[0]
        params = driver.execute_query.call_args.kwargs
        assert "UNWIND $uuids AS target_uuid" in query
        assert live_fact_predicate("e") in query
        assert "e.group_id = $group_id" in query
        assert query.rstrip().endswith("RETURN collect(DISTINCT e.uuid) AS live")
        assert "SET" not in query
        assert (params["uuids"], params["group_id"]) == (["a", "b", "c"], "user_x")

    @pytest.mark.asyncio
    async def test_no_row_means_none_is_live(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = ([], None, None)

        assert await live_fact_uuids(driver, "user_x", ["a"]) == set()

    @pytest.mark.asyncio
    async def test_a_failed_read_raises_for_the_caller(self) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = TimeoutError("reply lost")

        with pytest.raises(TimeoutError):
            await live_fact_uuids(driver, "user_x", ["a"])
