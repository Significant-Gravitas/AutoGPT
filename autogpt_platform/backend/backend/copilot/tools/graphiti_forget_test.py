"""Tests for the memory_forget tools and ``mark_edges_superseded``, which
shares their module.

The retraction itself (Cypher, per-uuid failures) is pinned in
``graphiti/recall_forget_test.py``; here the tools are exercised with that
layer either mocked or driven through a mock driver.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.edges import EntityEdge

from backend.copilot.graphiti import scope_lock
from backend.copilot.graphiti.memory_model import (
    ForgetResult,
    MemoryForgetFailure,
    MemoryForgetFailureCode,
)
from backend.copilot.graphiti.recall import live_fact_predicate
from backend.copilot.graphiti.recall_fake_redis import FakeRedis
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.model import ChatSession
from backend.copilot.tools.graphiti_forget import (
    _MAX_FAILURE_DETAIL,
    MemoryForgetConfirmTool,
    MemoryForgetSearchTool,
    _build_confirm_message,
    mark_edges_superseded,
)
from backend.copilot.tools.models import (
    MemoryForgetCandidatesResponse,
    MemoryForgetConfirmResponse,
)

_MODULE = "backend.copilot.tools.graphiti_forget"


async def _enabled(_user_id: str) -> bool:
    return True


@pytest.fixture(autouse=True)
def lock_redis(mocker) -> FakeRedis:
    """The real ``retract`` takes the graph's write lock; keep it in memory."""
    redis = FakeRedis()
    mocker.patch.object(
        scope_lock, "get_redis_async", mocker.AsyncMock(return_value=redis)
    )
    return redis


def _mock_driver(*results) -> AsyncMock:
    """A FalkorDB driver whose queries return ``results`` in order, for
    driving the real ``retract`` from the confirm tool."""
    driver = AsyncMock()
    driver.execute_query.side_effect = [(r, [], None) for r in results]
    return driver


class TestExpertMemoryScope:
    @pytest.mark.asyncio
    async def test_forget_search_uses_expert_memory_group(self) -> None:
        search_facts = AsyncMock(return_value=[])
        session = ChatSession.new("user-abc", dry_run=False, expert_id="expert-1")
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(f"{_MODULE}.search_facts", search_facts),
        ):
            await MemoryForgetSearchTool()._execute(
                "user-abc", session, query="private fact"
            )

        search_facts.assert_awaited_once_with(
            MemoryScope.for_expert("user-abc", "expert-1"), "private fact", limit=10
        )

    @pytest.mark.asyncio
    async def test_forget_confirm_uses_expert_memory_group(self) -> None:
        retract = AsyncMock(return_value=ForgetResult())
        session = ChatSession.new("user-abc", dry_run=False, expert_id="expert-1")
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(f"{_MODULE}.retract", retract),
        ):
            await MemoryForgetConfirmTool()._execute(
                "user-abc", session, uuids=["private-edge"]
            )

        retract.assert_awaited_once_with(
            MemoryScope.for_expert("user-abc", "expert-1"),
            ["private-edge"],
            hard=False,
        )


class TestForgetSearchCandidates:
    @pytest.mark.asyncio
    async def test_candidates_carry_uuid_fact_and_validity(self) -> None:
        edge = EntityEdge(
            uuid="e1",
            group_id="user_user-abc",
            source_node_uuid="a",
            target_node_uuid="b",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            name="works_on",
            fact="Alice works on Atlas",
            valid_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        session = ChatSession.new("user-abc", dry_run=False)
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(f"{_MODULE}.search_facts", AsyncMock(return_value=[edge])),
        ):
            response = await MemoryForgetSearchTool()._execute(
                "user-abc", session, query="atlas"
            )

        assert isinstance(response, MemoryForgetCandidatesResponse)
        assert response.candidates == [
            {
                "uuid": "e1",
                "fact": "Alice works on Atlas",
                "valid_from": "2025-01-01 00:00:00+00:00",
                "valid_to": "present",
            }
        ]


class TestForgetConfirmModes:
    @pytest.mark.asyncio
    async def test_hard_delete_asks_for_a_hard_retract(self) -> None:
        retract = AsyncMock(return_value=ForgetResult(deleted=["e1"]))
        session = ChatSession.new("user-abc", dry_run=False)
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(f"{_MODULE}.retract", retract),
        ):
            response = await MemoryForgetConfirmTool()._execute(
                "user-abc", session, uuids=["e1"], hard_delete=True
            )

        assert retract.await_args is not None
        assert retract.await_args.kwargs == {"hard": True}
        assert isinstance(response, MemoryForgetConfirmResponse)
        assert response.message == "1 memory edge(s) permanently deleted."

    @pytest.mark.asyncio
    async def test_unavailable_graph_is_an_error_response(self) -> None:
        session = ChatSession.new("user-abc", dry_run=False)
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(f"{_MODULE}.retract", AsyncMock(side_effect=OSError("refused"))),
        ):
            response = await MemoryForgetConfirmTool()._execute(
                "user-abc", session, uuids=["e1"]
            )

        assert not isinstance(response, MemoryForgetConfirmResponse)
        assert "temporarily unavailable" in response.message


class TestForgetFailuresAreActionable:
    """SECRT-2371: soft delete must not fail silently. Every failure — whether
    the query errored or matched nothing — has to carry a per-UUID reason so
    the model can act (retry, hard-delete, or tell the user) instead of seeing
    a bare "0 invalidated, N failed". Driven through the real ``retract``
    with a mock driver."""

    @pytest.mark.asyncio
    async def test_confirm_tool_reports_reasons_in_response(self) -> None:
        """End to end: a soft delete that matches nothing must return the
        per-UUID reason in both the structured `failures` field and the
        human-readable message — not a bare "0 invalidated, 1 failed"."""
        driver = _mock_driver([])  # the edge lookup finds nothing
        session = ChatSession.new("user-abc", dry_run=False)
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(
                "backend.copilot.graphiti.recall_forget.open_driver",
                MagicMock(return_value=driver),
            ),
        ):
            response = await MemoryForgetConfirmTool()._execute(
                "user-abc", session, uuids=["missing-uuid"]
            )

        assert isinstance(response, MemoryForgetConfirmResponse)
        assert response.deleted_uuids == []
        assert [f.uuid for f in response.failures] == ["missing-uuid"]
        assert response.failed_uuids == ["missing-uuid"]
        assert response.failures[0].code == MemoryForgetFailureCode.NO_MATCH
        # The reason must reach the model-visible message, not just the count.
        assert response.failures[0].reason in response.message
        assert "missing-uuid" in response.message

    @pytest.mark.asyncio
    async def test_confirm_tool_mixed_batch_reports_both(self) -> None:
        """A batch where some UUIDs delete and some fail must co-populate
        `deleted_uuids` and `failures`, and the message must carry BOTH the
        success count and the per-UUID failure detail."""
        driver = _mock_driver(
            [{"uuid": "kept"}],  # lookup: only "kept" exists
            [{"uuid": "kept"}],  # retract "kept"
            [],  # scrub its sentence
            [],  # find the entities to scrub (none)
            [],  # redact its episodes
        )
        session = ChatSession.new("user-abc", dry_run=False)
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(
                "backend.copilot.graphiti.recall_forget.open_driver",
                MagicMock(return_value=driver),
            ),
        ):
            response = await MemoryForgetConfirmTool()._execute(
                "user-abc", session, uuids=["kept", "gone"]
            )

        assert isinstance(response, MemoryForgetConfirmResponse)
        assert response.deleted_uuids == ["kept"]
        assert [f.uuid for f in response.failures] == ["gone"]
        # Two-part message: success half + failure detail half.
        assert "1 memory edge(s) retracted from memory." in response.message
        assert "1 failed" in response.message
        assert "gone" in response.message

    @pytest.mark.asyncio
    async def test_confirm_tool_reports_a_failed_clean_up(self) -> None:
        """Retracted, but the episode redaction failed: the model is told the
        fact is forgotten and that the clean-up did not finish."""
        driver = AsyncMock()
        driver.execute_query.side_effect = [
            ([{"uuid": "u1"}], [], None),  # lookup
            ([{"uuid": "u1"}], [], None),  # retract
            ([], [], None),  # scrub its sentence
            ([], [], None),  # find the entities to scrub (none)
            RuntimeError("down"),  # redact its episodes
        ]
        session = ChatSession.new("user-abc", dry_run=False)
        with (
            patch(f"{_MODULE}.is_enabled_for_user", _enabled),
            patch(
                "backend.copilot.graphiti.recall_forget.open_driver",
                MagicMock(return_value=driver),
            ),
        ):
            response = await MemoryForgetConfirmTool()._execute(
                "user-abc", session, uuids=["u1"]
            )

        assert isinstance(response, MemoryForgetConfirmResponse)
        assert response.deleted_uuids == ["u1"]
        assert [f.code for f in response.failures] == [
            MemoryForgetFailureCode.CLEANUP_ERROR
        ]
        assert "no longer recalled" in response.message
        assert "RuntimeError: down" in response.message


class TestBuildConfirmMessage:
    """`_build_confirm_message` formatting: the no-failure early return, the
    two-part success+failure string, and the bounded detail (thread: a driver
    outage failing every UUID must not blow the tool output past its size
    threshold and lose all detail)."""

    def test_no_failures_returns_summary_only(self) -> None:
        message = _build_confirm_message(3, "retracted from memory", [])
        assert message == "3 memory edge(s) retracted from memory."

    def test_caps_inlined_detail_and_notes_remainder(self) -> None:
        overflow = _MAX_FAILURE_DETAIL + 4
        failures = [
            MemoryForgetFailure(
                uuid=f"uuid-{i}",
                code=MemoryForgetFailureCode.NO_MATCH,
                reason="x" * 120,
            )
            for i in range(overflow)
        ]

        message = _build_confirm_message(0, "retracted from memory", failures)

        # Full count is reported, but only the first N reasons are inlined.
        assert f"{overflow} failed" in message
        assert f"…and {overflow - _MAX_FAILURE_DETAIL} more" in message
        assert message.count("uuid-") == _MAX_FAILURE_DETAIL
        # Bounded regardless of batch size — cannot grow with the input.
        assert len(message) < _MAX_FAILURE_DETAIL * 300


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

    @pytest.mark.asyncio
    async def test_expected_status_makes_the_write_conditional(self) -> None:
        """The ratification sweep's guard: only an edge still in that status
        and unexpired is touched, so a forget made since it was listed is
        kept, and the edge is reported failed."""
        driver = AsyncMock()
        driver.execute_query.return_value = ([], None, None)  # no longer tentative

        deleted, failed = await mark_edges_superseded(
            driver, ["u1"], reason="unratified", expected_status="tentative"
        )

        assert (deleted, failed) == ([], ["u1"])
        query = driver.execute_query.call_args.args[0]
        assert (
            "WHERE e.status = $expected_status AND e.expired_at IS NULL"
            " AND e.forgotten_at IS NULL" in query
        )
        assert driver.execute_query.call_args.kwargs["expected_status"] == "tentative"

    @pytest.mark.asyncio
    async def test_without_expected_status_only_a_live_fact_is_written(self) -> None:
        """The dream's demotions: an edge retired or forgotten since the dream
        read it is not overwritten, and is reported failed."""
        driver = AsyncMock()
        driver.execute_query.return_value = ([], None, None)  # forgotten meanwhile

        deleted, failed = await mark_edges_superseded(
            driver, ["u1"], reason="stale_fact"
        )

        assert (deleted, failed) == ([], ["u1"])
        query = driver.execute_query.call_args.args[0]
        assert f"WHERE {live_fact_predicate('e')}" in query
        assert "expected_status" not in driver.execute_query.call_args.kwargs
